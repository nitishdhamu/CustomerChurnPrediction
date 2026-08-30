import json
import os
from faker import Faker

def generate_diverse_name_pools():
    print("[*] Generating diverse international name pools...")
    
    # Strictly traditional English names (US and UK only) as requested
    locales = ['en_US', 'en_GB']
    fake = Faker(locales)
    
    first_names = set()
    last_names = set()
    
    # Generate 10000 times and rely on sets to deduplicate (prevents infinite loops if pool < 2000)
    for _ in range(10000):
        first_names.add(fake.first_name())
        last_names.add(fake.last_name())
            
    # Clean up any potential non-latin characters or weird formatting just in case
    # (Though we requested locales that usually provide localized names)
    # Actually, some locales return native scripts (like Hindi or Arabic). 
    # Let's ensure they are strings.
    first_names = list(first_names)
    last_names = list(last_names)
    
    data = {
        'first_names': first_names,
        'last_names': last_names
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/international_names.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        
    print(f"[+] Saved {len(first_names)} first names and {len(last_names)} last names.")
    print(f"[+] This enables {len(first_names) * len(last_names):,} unique combinations!")

if __name__ == "__main__":
    generate_diverse_name_pools()
