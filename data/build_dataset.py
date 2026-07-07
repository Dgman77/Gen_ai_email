"""
build_dataset.py
-----------------
Generates data/dataset.jsonl — a synthetic customer-support email dataset
for training/evaluating a Gen-AI email suggested-response system.

Design:
- 20 support categories x 5 variations each = 100 examples
- Each category has multiple email/reply templates + pools of names,
  order/invoice IDs, amounts, dates, products — randomly combined with a
  FIXED SEED so the output is diverse but 100% reproducible.
- Every row carries rich metadata (intent, entities, expected_actions, tone,
  difficulty, resolution_type, requires_followup, contains_multiple_issues)
  so the evaluator can score more than text similarity.
- No API key or network access required to build this dataset — it is fully
  offline and deterministic. (See generate_dataset_llm.py for an optional
  LLM-based alternative/expansion path using Gemini.)

Run:
    python build_dataset.py
Outputs:
    dataset.jsonl, train.jsonl, test.jsonl
"""

import json
import random
import uuid

random.seed(42)

NAMES = ["Priya Sharma", "James Okoro", "Wei Zhang", "Fatima Al-Sayed", "Carlos Mendes",
         "Emma Thompson", "Raj Patel", "Sofia Rossi", "Daniel Kim", "Aisha Bello",
         "Lucas Silva", "Grace Nakamura", "Mohammed Iqbal", "Olivia Brown", "Ivan Petrov",
         "Chidi Okafor", "Hana Kobayashi", "Diego Fernandez", "Anya Volkova", "Tom Fletcher"]

PRODUCTS = ["Pro Plan", "Team Workspace", "Enterprise Suite", "Starter Kit", "Premium Add-on"]

def rid(prefix):
    return f"{prefix}-{random.randint(10000, 99999)}"

def amt():
    return f"${random.choice([19.99, 29, 49.5, 99, 149, 12.75, 250, 8.99])}"

def date():
    return random.choice(["June 12", "June 28", "July 1", "July 3", "last Tuesday", "two weeks ago"])

TONES_POOL = ["neutral", "polite", "frustrated", "angry", "urgent", "confused", "grateful", "brief"]

# Each entry: (category, [ (email_template, reply_template, intent, expected_actions,
#              resolution_type, difficulty, tone_pool, multi_issue, followup) ... ])

CATS = {
"Billing Issue": [
 ("Hi team, I was charged {amt} for {product} but I thought my plan only cost {amt2}. Can you explain this charge? My account email is {name_l}.",
  "Hi {name}, thanks for flagging this. I checked your account and the {amt} charge reflects {product}, which includes an add-on beyond the base plan. I've attached a breakdown to your account dashboard so you can see exactly what's included. If anything still looks off after reviewing it, reply here and I'll loop in billing directly.",
  "billing clarification", ["Acknowledge", "Investigate charge", "Explain breakdown"], "Resolved - explanation given", "easy"),
 ("why was I charged {amt} again this month?? this is the SECOND time. fix this now.",
  "Hi {name}, I completely understand the frustration of seeing this a second time, and I'm sorry for the trouble. I've pulled your last two invoices and I'm investigating why the charge repeated. I'll have a definitive answer and, if it's an error, a refund processed within 24 hours. I'll follow up personally once I hear back from billing.",
  "billing dispute", ["Apologize", "Investigate charge", "Escalate to billing team"], "Escalated to billing team", "medium"),
],
"Refund Request": [
 ("I'd like a refund for invoice {inv}. I cancelled the {product} before the renewal date but was still charged {amt}.",
  "Hi {name}, thank you for reaching out, and I'm sorry for the inconvenience. I've confirmed invoice {inv} was charged after your cancellation date, which shouldn't have happened. I've submitted a full refund of {amt} — it should appear on your statement within 5-7 business days. Let me know if you don't see it by then.",
  "refund request", ["Apologize", "Verify cancellation date", "Issue refund"], "Refund issued", "medium"),
 ("refund please. order {order}. not what i expected.",
  "Hi there, thanks for letting us know. Could you share a bit more about what didn't match your expectations for order {order}? That'll help me process the right kind of refund or find a better fix for you — I want to make sure this is resolved properly.",
  "refund request", ["Acknowledge", "Ask for details", "Offer resolution"], "Pending more info", "hard"),
],
"Duplicate Charge": [
 ("I just noticed TWO charges of {amt} on {date} for the same {product}. This needs to be fixed immediately, I run a small business and this affects my cash flow.",
  "Hi {name}, I'm sorry for the stress this has caused, especially given the impact on your business. I can confirm both charges of {amt} on {date} were indeed duplicated due to a processing error on our end. I've already reversed the second charge — you should see it credited within 3-5 business days. Thank you for your patience.",
  "duplicate charge", ["Apologize", "Confirm duplicate", "Reverse charge"], "Refund issued", "medium"),
],
"Failed Payment": [
 ("My payment for invoice {inv} failed but I have sufficient funds. My account shows past due now, please help.",
  "Hi {name}, thanks for flagging this — I know a past-due flag can be stressful. I see the payment for invoice {inv} was declined by your card issuer, possibly due to a temporary hold on their end rather than an issue with your funds. Could you try updating your payment method or retrying the charge? I've paused any late fees on our side in the meantime.",
  "payment failure", ["Acknowledge", "Investigate decline reason", "Pause late fees"], "Pending customer action", "medium"),
],
"Subscription Cancellation": [
 ("please cancel my subscription. account: {name_l}. i no longer need this service.",
  "Hi {name}, I've gone ahead and cancelled your subscription effective at the end of your current billing cycle, so you'll retain access until then with no further charges. If you ever want to come back, your data will be saved for 30 days. Thanks for having been with us, and I'm happy to help with anything else in the meantime.",
  "cancellation", ["Confirm cancellation", "Explain access timeline", "Offer future re-activation info"], "Cancelled", "easy"),
 ("I want to cancel but I'm only doing it because the new pricing is way too expensive for what I get. Just so you know.",
  "Hi {name}, thank you for the honest feedback, and I'm sorry the new pricing doesn't feel like the right value for you. I've processed your cancellation, effective at the end of this billing period. I've also shared your feedback with our pricing team since it genuinely helps shape future changes. If you'd like, I can check whether a lower-tier plan might still suit your needs before you go.",
  "cancellation with complaint", ["Acknowledge feedback", "Confirm cancellation", "Offer alternative plan"], "Cancelled", "hard"),
],
"Upgrade/Downgrade": [
 ("Can I upgrade from {product} to Enterprise Suite mid-cycle? Will I be charged the full price again?",
  "Hi {name}, great question! Yes, you can upgrade anytime — we'll prorate the difference so you're only charged for the remaining days in your current cycle, not the full Enterprise price. I can process this upgrade right now if you'd like, or you're welcome to do it yourself from Settings > Plan.",
  "upgrade inquiry", ["Answer question", "Explain proration", "Offer to process"], "Resolved - info given", "easy"),
],
"Account Locked": [
 ("My account got locked after 3 failed login attempts, but I KNOW my password is correct. This is urgent, I have a client demo in an hour.",
  "Hi {name}, I understand the urgency given your demo — let's get this sorted right away. I've unlocked your account just now; please try logging in again. If it locks again, it may be a cached old password on one of your devices, so try clearing that or using an incognito window. Reply immediately if you're still stuck and I'll stay on this with you.",
  "account lockout", ["Apologize", "Unlock account", "Provide workaround"], "Resolved - account unlocked", "hard"),
],
"Login Problems": [
 ("i cant login. keeps saying invalid. tried 5 times",
  "Hi there, sorry for the trouble logging in. Could you confirm the email address you're using to sign in? In the meantime, try resetting your password using the 'Forgot Password' link on the login page — that resolves this for most users. Let me know if it still doesn't work and I'll dig deeper on our end.",
  "login issue", ["Acknowledge", "Ask for details", "Suggest password reset"], "Pending customer action", "medium"),
],
"Password Reset": [
 ("I haven't received the password reset email. Checked spam too. Can you help?",
  "Hi {name}, sorry about that! I've manually triggered a new password reset email to your registered address — please check both inbox and spam again in the next few minutes. If it still doesn't arrive, let me know and I can verify the email on file or reset it through a different method.",
  "password reset", ["Acknowledge", "Resend reset email", "Offer alternative verification"], "Resolved - email resent", "easy"),
],
"MFA Problems": [
 ("Lost my phone so I can't get my MFA codes anymore. Locked out of my account with all my client data. Please help ASAP, this is critical for my business.",
  "Hi {name}, I completely understand how critical this is, and I'm sorry for the stress. For security, MFA recovery requires identity verification — I've sent you a secure verification link to your registered email that will let us temporarily disable MFA once confirmed. I'll personally monitor this and follow up within the hour to make sure you're back in.",
  "MFA recovery", ["Apologize", "Verify identity", "Temporarily disable MFA"], "Escalated - pending verification", "hard"),
],
"Shipping Delay": [
 ("Order {order} was supposed to arrive by {date} and it's still not here. Can you check the status?",
  "Hi {name}, thanks for checking in, and sorry for the delay on order {order}. I checked with our carrier and it's currently held at a regional facility, with a new estimated delivery of 2-3 additional days. I'll keep an eye on it and reach out proactively if anything changes — thank you for your patience.",
  "shipping delay", ["Acknowledge", "Check carrier status", "Provide new estimate"], "Pending delivery", "medium"),
],
"Missing Package": [
 ("Tracking says my package for order {order} was delivered on {date} but I never received it. I've checked with neighbors too.",
  "Hi {name}, I'm sorry to hear that — that's frustrating, especially after checking around already. I've filed a claim with the carrier for order {order} and, since this typically takes a couple of days to investigate, I'm also going ahead and shipping a free replacement now so you're not left waiting. You'll get tracking details shortly.",
  "missing package", ["Apologize", "File carrier claim", "Ship replacement"], "Replacement shipped", "hard"),
],
"Wrong Item Received": [
 ("I ordered the blue {product} but received a red one instead. Order {order}.",
  "Hi {name}, thanks for letting us know, and sorry for the mix-up on order {order}. I've arranged for the correct blue item to be shipped right away at no extra cost, and you don't need to return the red one — please keep or donate it. You should receive tracking for the replacement within 24 hours.",
  "wrong item", ["Apologize", "Ship correct item", "Waive return requirement"], "Replacement shipped", "easy"),
],
"Damaged Product": [
 ("The {product} I received today arrived cracked. Very disappointed, this was a gift for my daughter's birthday tomorrow.",
  "Hi {name}, I'm so sorry — I know how much timing matters here with tomorrow's birthday. I've arranged an expedited replacement to arrive by tomorrow morning where possible, at no additional cost, and there's no need to return the damaged item. I'll send tracking as soon as it's available and I really hope this doesn't affect the celebration.",
  "damaged product", ["Apologize", "Expedite replacement", "Waive return"], "Replacement expedited", "hard"),
],
"Return Request": [
 ("I want to return the {product} I bought, order {order}. It just didn't fit my needs.",
  "Hi {name}, no problem at all — I've started a return for order {order}. You'll receive a prepaid shipping label by email within the hour; once we receive the item, your refund will be processed within 3-5 business days. Let me know if you'd like help finding an alternative that better fits your needs instead.",
  "return request", ["Acknowledge", "Provide return label", "Explain refund timeline"], "Return initiated", "easy"),
],
"Technical Bug": [
 ("The export button on the dashboard does nothing when I click it. Using Chrome, latest version. This is blocking my monthly report due today.",
  "Hi {name}, thanks for the detailed report, and sorry this is blocking your report deadline today. This looks like a known issue affecting exports in the latest Chrome update, and our engineering team is actively working on a fix. As a workaround, exporting via Firefox or using the API endpoint should work in the meantime. I'll update you the moment the fix ships.",
  "bug report", ["Acknowledge", "Confirm known issue", "Provide workaround"], "Workaround provided, escalated to engineering", "hard"),
],
"Feature Request": [
 ("Would love if you could add dark mode to the mobile app. Any plans for this?",
  "Hi {name}, thanks for the suggestion — dark mode is one of our most requested features and it's currently on our roadmap. I don't have an exact release date yet, but I've added your vote to the request, which helps us prioritize. I'll make sure you're notified as soon as it ships.",
  "feature request", ["Acknowledge", "Confirm roadmap status", "Log request"], "Logged for roadmap", "easy"),
],
"Integration Problem": [
 ("Our Slack integration stopped posting notifications three days ago. No error messages, it just silently stopped. We rely on this heavily for our support team.",
  "Hi {name}, thank you for flagging this, and I understand how disruptive a silent failure like this is for your team. I've checked our logs and see the Slack webhook token expired on our end without triggering an alert — that's on us, and I've already reissued a fresh token and reconnected the integration. Please confirm on your end that notifications are flowing again; I'll also make sure we add expiry alerts so this doesn't happen silently again.",
  "integration failure", ["Apologize", "Diagnose root cause", "Fix and confirm"], "Resolved - integration restored", "hard"),
],
"Positive Feedback": [
 ("Just wanted to say your support team helped me out yesterday and it was honestly the best support experience I've had in years. Thank you!",
  "Hi {name}, thank you so much for taking the time to share this — it genuinely made our day! I'll be sure to pass your kind words along to the specific teammate who helped you. Please don't hesitate to reach out anytime, and thanks again for being a great customer.",
  "positive feedback", ["Thank customer", "Pass along feedback"], "Acknowledged", "easy"),
],
"Escalation Request": [
 ("This is my third email about the same billing issue with no resolution. I want to speak to a manager immediately.",
  "Hi {name}, I sincerely apologize that this has taken three emails without resolution — that's not the experience we want for you, and I understand the frustration. I've escalated your case directly to our billing team lead, who will reach out within 4 hours with a concrete resolution. I've also flagged the prior history so you won't need to re-explain anything.",
  "escalation", ["Apologize sincerely", "Escalate to manager", "Reference prior history"], "Escalated to manager", "hard"),
],
}

def build():
    rows = []
    idx = 1
    for category, templates in CATS.items():
        # cycle through templates to get 5 examples per category
        for i in range(5):
            email_t, reply_t, intent, actions, resolution, base_difficulty = templates[i % len(templates)]
            name = random.choice(NAMES)
            product = random.choice(PRODUCTS)
            order = rid("ORD")
            inv = rid("INV")
            amount = amt()
            amount2 = amt()
            dt = date()

            fill = dict(name=name, name_l=name.split()[0].lower(), product=product,
                        order=order, inv=inv, amt=amount, amt2=amount2, date=dt)

            email = email_t.format(**fill)
            reply = reply_t.format(**fill)

            entities = [v for k, v in [("product", product), ("order", order),
                        ("invoice", inv), ("amount", amount), ("date", dt)]
                        if v in email or v in reply]

            tone = random.choice(TONES_POOL)
            difficulty = base_difficulty if i < 3 else random.choice(["easy", "medium", "hard"])
            multi_issue = i == 4 and random.random() < 0.3
            followup = difficulty == "hard" or random.random() < 0.25

            rows.append({
                "id": f"{idx:04d}",
                "category": category,
                "difficulty": difficulty,
                "customer_email": email,
                "ideal_agent_reply": reply,
                "intent": intent,
                "entities": entities,
                "expected_actions": actions,
                "tone": tone,
                "resolution_type": resolution,
                "requires_followup": followup,
                "contains_multiple_issues": multi_issue,
                "language": "English",
                "source": "template_synthetic",
            })
            idx += 1

    # Hand-crafted edge cases (explicitly labelled)
    edge_cases = [
        {
            "id": f"{idx:04d}", "category": "Billing Issue", "difficulty": "hard",
            "customer_email": "hi i pay too much money every month and nobody help me i am very sad customer since 2 year pls somebody read this",
            "ideal_agent_reply": "Hi there, I'm really sorry to hear you've felt unheard for so long — that's not okay, and I want to personally make this right. Could you share your account email so I can pull up your billing history and see exactly what you're being charged and why? Once I have that, I'll walk you through everything clearly and see if a better plan fits your usage.",
            "intent": "billing dissatisfaction", "entities": [], "expected_actions": ["Apologize sincerely", "Ask for account details", "Offer plan review"],
            "tone": "broken_english_frustrated", "resolution_type": "Pending more info", "requires_followup": True,
            "contains_multiple_issues": False, "language": "English", "source": "hand_crafted",
        },
        {
            "id": f"{idx+1:04d}", "category": "Return Request", "difficulty": "medium",
            "customer_email": "return",
            "ideal_agent_reply": "Hi there, happy to help with a return! Could you share your order number and the reason for the return? Once I have that, I can send over a prepaid return label and get your refund started right away.",
            "intent": "return request", "entities": [], "expected_actions": ["Ask for order number", "Ask for reason"],
            "tone": "brief", "resolution_type": "Pending more info", "requires_followup": True,
            "contains_multiple_issues": False, "language": "English", "source": "hand_crafted",
        },
        {
            "id": f"{idx+2:04d}", "category": "Escalation Request", "difficulty": "hard",
            "customer_email": "I've now emailed support 4 times about (1) a duplicate charge of $89, (2) my account being locked, and (3) a missing refund from three weeks ago. Nobody has resolved ANY of this. I am extremely close to cancelling and posting about this publicly. I need someone senior to handle this TODAY.",
            "ideal_agent_reply": "Hi, I want to sincerely apologize — three unresolved issues across four emails is completely unacceptable, and I understand why you're at your limit. I'm personally taking ownership of all three items right now: the duplicate $89 charge, the account lock, and the missing refund. I'll have concrete updates on each within 2 hours today, and I've escalated this to my team lead as well so you have a direct line if anything stalls.",
            "intent": "multi-issue escalation", "entities": ["$89"], "expected_actions": ["Apologize sincerely", "Take ownership of all issues", "Escalate", "Commit to timeline"],
            "tone": "angry", "resolution_type": "Escalated - owner assigned", "requires_followup": True,
            "contains_multiple_issues": True, "language": "English", "source": "hand_crafted",
        },
        {
            "id": f"{idx+3:04d}", "category": "Positive Feedback", "difficulty": "easy",
            "customer_email": "quick note - the new dashboard update is SO much faster. whatever you did, keep doing it :)",
            "ideal_agent_reply": "Hi, thank you for the quick note — this really made our day! I'll pass this along to the engineering team who worked hard on the performance improvements. Feel free to reach out anytime if you have more feedback or run into anything.",
            "intent": "positive feedback", "entities": [], "expected_actions": ["Thank customer", "Pass along feedback"],
            "tone": "grateful", "resolution_type": "Acknowledged", "requires_followup": False,
            "contains_multiple_issues": False, "language": "English", "source": "hand_crafted",
        },
    ]
    rows.extend(edge_cases)
    return rows


def main():
    rows = build()
    random.shuffle(rows)

    with open("dataset.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    split_idx = int(len(rows) * 0.8)
    train, test = rows[:split_idx], rows[split_idx:]

    with open("train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open("test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

    print(f"Built {len(rows)} examples -> dataset.jsonl")
    print(f"Split: {len(train)} train / {len(test)} test")


if __name__ == "__main__":
    main()
