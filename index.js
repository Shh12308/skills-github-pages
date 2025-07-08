import express from "express";
import bodyParser from "body-parser";
import admin from "firebase-admin";
import jwt from "jsonwebtoken"; // for Apple notifications
import dotenv from "dotenv";

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());

// === Initialize Firebase Admin SDK ===
const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || "{}");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();
const usersRef = db.collection("users");

// === 🍋 Lemon Squeezy Webhook ===
app.post("/webhook/lemonsqueezy", async (req, res) => {
  const { meta, data } = req.body;
  const event = meta?.event_name;

  if (!event.includes("subscription")) return res.sendStatus(204);

  try {
    const email = data.attributes.user_email;
    const productName = data.attributes.product_name.toLowerCase();

    const plan = productName.includes("ultimate")
      ? "ultimate"
      : productName.includes("pro")
      ? "pro"
      : "free";

    const match = await usersRef.where("profile.email", "==", email).get();
    if (match.empty) return res.status(404).send("❌ No user found for email");

    match.forEach((doc) => {
      doc.ref.update({
        plan,
        subscribedAt: new Date().toISOString(),
        billingProvider: "lemon_squeezy"
      });
    });

    console.log(`✅ LemonSqueezy → ${email} upgraded to ${plan}`);
    res.sendStatus(200);
  } catch (err) {
    console.error("Lemon webhook error:", err);
    res.sendStatus(500);
  }
});

// === 🍎 Apple App Store Webhook ===
app.post("/webhook/apple", async (req, res) => {
  try {
    const signedPayload = req.body.signedPayload;
    const decoded = jwt.decode(signedPayload, { complete: true });
    const notification = decoded?.payload;

    const appleUserId = notification.data?.appAccountToken; // Should be Firebase UID
    const productId = notification.data?.productId;

    const plan = productId.includes("ultimate")
      ? "ultimate"
      : productId.includes("pro")
      ? "pro"
      : "free";

    const match = await usersRef.where("appleId", "==", appleUserId).get();
    if (match.empty) return res.status(404).send("❌ Apple user not found");

    match.forEach((doc) => {
      doc.ref.update({
        plan,
        subscribedAt: new Date().toISOString(),
        billingProvider: "apple"
      });
    });

    console.log(`🍎 Apple → ${appleUserId} to ${plan}`);
    res.sendStatus(200);
  } catch (err) {
    console.error("Apple webhook error:", err);
    res.sendStatus(500);
  }
});

// === 🤖 Google Play Webhook ===
app.post("/webhook/google", async (req, res) => {
  try {
    const message = JSON.parse(
      Buffer.from(req.body.message.data, "base64").toString()
    );
    const notification = message.subscriptionNotification;

    const productId = notification.subscriptionId;
    const purchaseToken = notification.purchaseToken;

    const plan = productId.includes("ultimate")
      ? "ultimate"
      : productId.includes("pro")
      ? "pro"
      : "free";

    const match = await usersRef
      .where("googlePurchaseToken", "==", purchaseToken)
      .get();

    if (match.empty) return res.status(404).send("❌ Google user not found");

    match.forEach((doc) => {
      doc.ref.update({
        plan,
        subscribedAt: new Date().toISOString(),
        billingProvider: "google"
      });
    });

    console.log(`🤖 Google → ${purchaseToken} to ${plan}`);
    res.sendStatus(200);
  } catch (err) {
    console.error("Google webhook error:", err);
    res.sendStatus(500);
  }
});

// === Server Start ===
app.listen(PORT, () => {
  console.log(`🚀 Zynara Webhook listening on port ${PORT}`);
});
