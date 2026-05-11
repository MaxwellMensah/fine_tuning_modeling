import json
import random

# expanded datasets for better variety
locations = [
    "London, UK", "Lagos, Nigeria", "New York, USA", "Dubai, UAE", 
    "Singapore", "Moscow, Russia", "Tokyo, Japan", "Paris, France", 
    "Sao Paulo, Brazil", "Sydney, Australia", "Berlin, Germany"
]
ips = [
    "192.168.1.1", "45.76.12.3", "103.22.11.5", "88.150.12.9", 
    "Unknown", "VPN_Detected", "Tor_Exit_Node", "Residential_Proxy"
]
times = [
    "2:15 AM", "3:05 AM", "11:45 PM", "1:20 PM", "9:00 AM", 
    "4:30 AM", "12:00 PM", "6:15 PM", "10:00 PM"
]
amounts = [15, 50, 120, 550, 1200, 2500, 4800, 7500, 12000, 25000]
currencies = ["USD", "EUR", "GBP", "NGN", "AED"]

def generate_fraud_data(num_records=500):
    dataset = []
    
    for i in range(num_records):
        loc = random.choice(locations)
        ip = random.choice(ips)
        time = random.choice(times)
        amt = random.choice(amounts)
        cur = random.choice(currencies)
        
        # Complex Logic for "Is Suspicious"
        # Suspicious if: High amount, bad IP, or late night + high amount
        is_bad_ip = ip in ["Unknown", "VPN_Detected", "Tor_Exit_Node"]
        is_late_night = any(x in time for x in ["AM", "PM"]) and ("2" in time or "3" in time or "4" in time)
        
        is_suspicious = is_bad_ip or (amt > 5000) or (is_late_night and amt > 1000)
        
        user_query = f"Analyze this transaction: User from {loc} attempting a payment of {amt} {cur} using IP {ip} at {time}."
        
        if is_suspicious:
            reasoning = [
                f"Step 1: Context - The user is attempting a {cur} transaction from {loc}.",
                f"Step 2: Anomaly Detection - Found high-risk indicators: { 'Flagged IP ('+ip+')' if is_bad_ip else 'Unusual timing ('+time+')' } and {'Large transaction volume' if amt > 5000 else 'Suspicious amount for time of day'}.",
                f"Step 3: Risk Assessment - The probability of account takeover or credit card fraud is elevated.",
                f"Conclusion: High Risk. Transaction blocked and flagged for manual review."
            ]
        else:
            reasoning = [
                f"Step 1: Context - User is initiating a standard {amt} {cur} payment.",
                f"Step 2: Anomaly Detection - IP {ip} appears clean and the time {time} aligns with typical user behavior for {loc}.",
                f"Step 3: Risk Assessment - No indicators of fraud, velocity limits, or suspicious routing detected.",
                f"Conclusion: Low Risk. Transaction approved."
            ]

        # NEW FORMAT: ChatML / Llama 3 Conversational Format
        record = {
            "conversations": [
                {"role": "system", "content": "You are a fraud detection expert. Analyze transactions using step-by-step reasoning."},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": " ".join(reasoning)}
            ]
        }
        dataset.append(record)

    with open("fraud_dataset.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Created {num_records} records in ChatML format.")

generate_fraud_data(500)