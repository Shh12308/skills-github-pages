// === DEPENDENCIES ===
import express from "express";
import bodyParser from "body-parser";
import admin from "firebase-admin";
import jwt from "jsonwebtoken"; // For Apple webhooks
import dotenv from "dotenv";

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());

// === FIREBASE ADMIN INIT ===
const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || "{}");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();
const usersRef = db.collection("users");

// === 🍋 LEMON SQUEEZY WEBHOOK ===
app.post("/webhook/lemonsqueezy", async (req, res) => {
  const { meta, data } = req.body;
  const event = meta?.event_name;

  console.log(`📩 LemonSqueezy Webhook received: ${event}`);

  if (!event.includes("subscription")) return res.sendStatus(204);

  try {
    const email = data.attributes.user_email?.toLowerCase().trim();
    const productName = data.attributes.product_name.toLowerCase();

    const plan = productName.includes("ultimate")
      ? "ultimate"
      : productName.includes("pro")
      ? "pro"
      : "free";

    const snapshot = await usersRef.where("profile.email", "==", email).get();

    if (snapshot.empty) {
      console.log(`⚠️ No user found for ${email}. Creating placeholder user.`);
      await usersRef.doc(email).set({
        plan,
        profile: { email },
        subscribedAt: new Date().toISOString(),
        billingProvider: "lemon_squeezy"
      });
      return res.status(200).send("✅ User created and plan set");
    }

    snapshot.forEach(doc => {
      doc.ref.update({
        plan,
        subscribedAt: new Date().toISOString(),
        billingProvider: "lemon_squeezy"
      });
    });

    console.log(`✅ ${email} upgraded to ${plan}`);
    res.sendStatus(200);
  } catch (err) {
    console.error("❌ LemonSqueezy webhook error:", err.message);
    res.sendStatus(500);
  }
});

// === 🍎 APPLE APP STORE WEBHOOK ===
app.post("/webhook/apple", async (req, res) => {
  try {
    const signedPayload = req.body.signedPayload;
    const decoded = jwt.decode(signedPayload, { complete: true });
    const notification = decoded?.payload;

    const appleUserId = notification.data?.appAccountToken; // Firebase UID
    const productId = notification.data?.productId;

    const plan = productId.includes("ultimate")
      ? "ultimate"
      : productId.includes("pro")
      ? "pro"
      : "free";

    const snapshot = await usersRef.where("appleId", "==", appleUserId).get();

    if (snapshot.empty) return res.status(404).send("❌ Apple user not found");

    snapshot.forEach(doc => {
      doc.ref.update({
        plan,
        subscribedAt: new Date().toISOString(),
        billingProvider: "apple"
      });
    });

    console.log(`🍎 Apple → ${appleUserId} upgraded to ${plan}`);
    res.sendStatus(200);
  } catch (err) {
    console.error("❌ Apple webhook error:", err.message);
    res.sendStatus(500);
  }
});

// === 🤖 GOOGLE PLAY WEBHOOK ===
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

    const snapshot = await usersRef
      .where("googlePurchaseToken", "==", purchaseToken)
      .get();

    if (snapshot.empty) return res.status(404).send("❌ Google user not found");

    snapshot.forEach(doc => {
      doc.ref.update({
        plan,
        subscribedAt: new Date().toISOString(),
        billingProvider: "google"
      });
    });

    console.log(`🤖 Google → ${purchaseToken} upgraded to ${plan}`);
    res.sendStatus(200);
  } catch (err) {
    console.error("❌ Google webhook error:", err.message);
    res.sendStatus(500);
  }
});

// === START SERVER ===
app.listen(PORT, () => {
  console.log(`🚀 Zynara Webhook listening on port ${PORT}`);
});
