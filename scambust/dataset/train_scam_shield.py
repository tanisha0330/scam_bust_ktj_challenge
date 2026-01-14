import pandas as pd
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os
import pickle  # Model save karne ke liye

# ---------------------------------------------------------
# INSTRUCTIONS (Hindi mein):
# 1. Terminal open karein aur likhein: pip install pandas scikit-learn
# 2. Phir code run karein: python train_scam_shield.py
# ---------------------------------------------------------

# ---------------------------------------------------------
# STEP 1: DATA LOAD KARNA
# ---------------------------------------------------------

file_name = "public_sms.csv"

# Check karte hain ki kya aapne dataset download karke folder mein rakha hai?
if os.path.exists(file_name):
    print(f"✅ Asli file '{file_name}' mil gayi! Data wahin se load ho raha hai...")
    # Asli file se data load karna
    df = pd.read_csv(file_name)
    
    # NOTE: Agar aapki csv file mein columns ke naam alag hain, toh rename karein.
    # Niche wali line check kar lena dataset ke hisaab se:
    # df = df.rename(columns={'v2': 'message_text', 'v1': 'is_scam'})

else:
    print("⚠️ File nahi mili. Koi baat nahi! Hum 'Mock Data' use kar rahe hain seekhne ke liye.")
    
    csv_data = """sms_id,message_text,is_scam
SMS001,TRAI verification pending. Confirm details by calling +91-98728-31517.,1
SMS002,Address incomplete. Reschedule delivery: https://rebrand.ly/GdfOiZ6 Jaldi karo.,1
SMS003,PhonePe Security: Malware detected. Install Remote Assist from https://cybercrime-case.in,1
SMS004,Hi, I’ll reach by 7:00 PM. See you soon.,0
SMS005,Device warranty expiring today. Verify at Jaldi karo.,1
SMS006,URGENT: Jio password change hua. Agar aapne nahi kiya to reset: https://amazon.in,0
SMS007,Apple Security: Malware detected. Install QuickSupport from https://verify-now.co,1
SMS008,BESCOM: Bill overdue ₹7500. Power cut in 2 hours. Pay at https://bit.ly/effIp1B,1
SMS009,OLX: Buyer sent payment. To receive ₹50000, share OTP.,1
SMS010,Delhivery: Delivery attempt failed. Reschedule at https://t.co/8Kjuhfj,1
SMS011,KYC update required. Share PAN/Aadhaar photo on WhatsApp.,1
SMS012,Prize release requires processing fee ₹99. Pay via UPI.,1
SMS013,Dear customer, your card will be blocked today. Verify details at https://rebrand.ly,1
SMS014,Reminder: Appointment with Dr. Mehta on 03-Nov at 7:00 PM.,0
SMS015,Hi Mom, new number. Emergency. Send ₹1500 to UPI.,1
SMS016,Blue Dart: Delivery attempt failed. Reschedule at https://rebrand.ly,1
SMS017,Hi, I’ll reach by 10:00 AM. See you soon.,0
SMS018,HDFC: Rs.50000 debited from A/C XX8082. Avl bal: Rs.45000.,0
SMS019,DTDC: Delivery attempt fail. Reschedule: https://tinyurl.com,1
SMS020,Earn ₹100000/day. Join VIP group using https://bit.ly,1
"""
    df = pd.read_csv(io.StringIO(csv_data))

print("--- Data ki pehli 5 lines ---")
print(df.head())
print(f"\nTotal messages loaded: {len(df)}")

# ---------------------------------------------------------
# STEP 2: MODEL TRAINING (Machine ko sikhana)
# ---------------------------------------------------------

if 'message_text' not in df.columns:
    print("\n❌ Error: CSV file mein 'message_text' column nahi mila.")
    # Hack: Agar column ka naam kuch aur hai, toh hum pehla text column dhoond lete hain
    text_cols = [col for col in df.columns if 'text' in col.lower() or 'msg' in col.lower() or 'message' in col.lower()]
    if text_cols:
        print(f"Shayad aapka column '{text_cols[0]}' hai. Hum ise use kar rahe hain.")
        X = df[text_cols[0]]
    else:
        print("Column nahi mila. Code band ho raha hai.")
        exit()
else:
    X = df['message_text']

# Target column dhoondna (is_scam, label, etc)
if 'is_scam' in df.columns:
    y = df['is_scam']
elif 'label' in df.columns:
    y = df['label']
else:
    print("Target column (is_scam/label) nahi mila.")
    exit()

print("\n🤖 Model training shuru ho rahi hai...")

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)
print("✅ Model training complete!")

# ---------------------------------------------------------
# STEP 3: MODEL SAVING (Dimaag ko save karna)
# ---------------------------------------------------------
model_filename = 'scam_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"💾 Model ko '{model_filename}' naam se save kar diya gaya hai.")
print("Ab aap is file ko App ya Website mein use kar sakte hain.")

# ---------------------------------------------------------
# STEP 4: TESTING
# ---------------------------------------------------------

def check_scam(text_message):
    prediction = model.predict([text_message])[0]
    if prediction == 1:
        return "🚨 SCAM ALERT! (Savdhaan Rahein)"
    else:
        return "✅ Safe Message"

print("\n--- Live Testing Results (Naye Messages) ---")
test_messages = [
    "Tera bhai ghar pahuch gaya hai.", 
    "Urgent! Your electricity will be cut. Pay 500rs at link http://fake.com",
    "Hello mom, send me money urgent UPI.",
    "Your appointment is confirmed for Monday." 
]

for msg in test_messages:
    result = check_scam(msg)
    print(f"Message: '{msg}'\nResult: {result}\n")

print("--- Apna khud ka message check karein ---")
try:
    user_msg = input("Koi message type karein aur Enter dabayein: ")
    if user_msg:
        print(f"Result: {check_scam(user_msg)}")
except:
    pass