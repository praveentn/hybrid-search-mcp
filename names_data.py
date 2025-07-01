# names_data.py
# Indian and Kerala names with funny one-liners
NAMES_ONELINERS = {
    # Common Indian Male Names
    "raj": "Raj thinks he's the king, but he can't even rule his own remote control!",
    "arjun": "Arjun aims for the stars but usually hits the neighbor's window instead.",
    "vikram": "Vikram is so brave, he once fought a mosquito for 3 hours... and lost.",
    "rohit": "Rohit is like a good WiFi connection - everyone wants him, but he's rarely available.",
    "amit": "Amit has a photographic memory, unfortunately it's out of film most of the time.",
    "suresh": "Suresh is so organized, he color-codes his chaos.",
    "ramesh": "Ramesh thinks outside the box because someone locked him out of it.",
    "kiran": "Kiran shines bright like a diamond... that's been through a blender.",
    "deepak": "Deepak lights up every room he enters, mainly because he forgets to turn off the lights.",
    "anand": "Anand finds joy in everything, including his own bad jokes.",
    
    # Common Indian Female Names  
    "priya": "Priya is so sweet, even her WiFi password is 'honey123'.",
    "anjali": "Anjali dances to the beat of her own drum... which is always out of tune.",
    "kavya": "Kavya writes poetry so beautiful, it makes onions cry tears of joy.",
    "meera": "Meera sings so well, even the neighbors' dogs request encores.",
    "shreya": "Shreya is blessed with everything... except the ability to find matching socks.",
    "neha": "Neha has eyes like stars - they only come out at night and confuse everyone.",
    "pooja": "Pooja worships coffee more than any deity, and her prayers are always answered.",
    "ritu": "Ritu changes faster than seasons, but somehow always looks fabulous.",
    "sunita": "Sunita is so bright, she needs her own solar panel warning label.",
    "geeta": "Geeta's life is like a song - beautiful, but nobody can understand the lyrics.",
    
    # Kerala Specific Names (Male)
    "ravi": "Ravi shines brighter than coconut oil on a summer day in Kerala.",
    "unni": "Unni is small in size but his appetite for banana chips is legendary.",
    "babu": "Babu talks so much, even the backwaters of Kerala ask him to keep it down.",
    "sethu": "Sethu builds bridges everywhere he goes, mainly because he burns them first.",
    "appu": "Appu is sweeter than payasam, but twice as sticky when it comes to favors.",
    "mani": "Mani collects everything like precious gems, including expired coupons.",
    "chettan": "Chettan is everyone's big brother, whether they want one or not.",
    "dasan": "Dasan serves others so well, mosquitoes have appointed him their official blood donor.",
    "krishnan": "Krishnan plays the flute of life, but it always sounds like a broken car horn.",
    "balan": "Balan is strong like an elephant, but gentle like a butterfly with anger issues.",
    
    # Kerala Specific Names (Female)
    "radha": "Radha loves so deeply, even her plants get jealous of her attention to others.",
    "maya": "Maya creates illusions so real, she once convinced herself she was on time.",
    "leela": "Leela's play is divine, especially when she pretends to understand cricket.",
    "suma": "Suma is a good flower who blooms even in the toughest Malayalam movie plots.",
    "sita": "Sita is so pure, she makes coconut water feel guilty about its mild flavor.",
    "ganga": "Ganga flows through life with grace, except when stuck in Kochi traffic.",
    "lakshmi": "Lakshmi brings wealth and prosperity... mainly to the local sweet shops.",
    "parvati": "Parvati has the power to move mountains, but can't find her car keys.",
    "saraswati": "Saraswati is blessed with knowledge, which she uses primarily for WhatsApp forwards.",
    "devi": "Devi is a goddess among mortals, but still can't fold a fitted sheet properly.",
    
    # More Common Names
    "arun": "Arun rises early like the sun, then immediately regrets it like everyone else.",
    "vinod": "Vinod spreads joy like butter on toast - unevenly and with too much enthusiasm.",
    "mukesh": "Mukesh has a face that could launch a thousand ships... back to shore.",
    "sandra": "Sandra is sandy like a beach - beautiful but gets everywhere and is hard to clean up.",
    "latha": "Latha climbs the vine of success, but always gets distracted by the flowers.",
    "sindhu": "Sindhu flows like a river of wisdom... that occasionally floods with confusion.",
    "bindu": "Bindu is a point of light in everyone's life, even if she's often dim.",
    "mini": "Mini is small but mighty, like a samosa that packs surprising heat.",
    "shiny": "Shiny sparkles wherever she goes, mainly due to her excessive use of glitter.",
    "jinu": "Jinu wins hearts faster than a Kerala lottery ticket wins hopes and dreams."
}

def get_oneliner(name):
    """Get a funny one-liner for a given name (case-insensitive)"""
    name_lower = name.lower().strip()
    return NAMES_ONELINERS.get(name_lower, f"I don't have a one-liner for {name}, but I'm sure they're awesome anyway!")

def get_all_names():
    """Get list of all available names"""
    return list(NAMES_ONELINERS.keys())
