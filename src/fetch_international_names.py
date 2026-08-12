import json
import os
from faker import Faker

def generate_diverse_name_pools():
    print("[*] Generating diverse international name pools...")
    
    # Locales for USA, India, China (Pinyin), Spain, France, Germany, Japan (Romaji), Arabic (Latin)
    locales = ['en_US', 'hi_IN', 'zh_CN', 'es_ES', 'fr_FR', 'de_DE', 'ja_JP', 'ar_AE']
    fake = Faker(locales)
    
    first_names = set()
    last_names = set()
    
    # Generate until we have 2000 of each
    while len(first_names) < 2000 or len(last_names) < 2000:
        if len(first_names) < 2000:
            first_names.add(fake.first_name())
        if len(last_names) < 2000:
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
