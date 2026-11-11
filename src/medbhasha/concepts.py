"""Small bilingual clinical concept table used by the retriever.

Each line ``canonical: variant | variant | ...`` lists synonyms, English / Bengali / abbreviation
variants and drug-class members that should count as the *same concept* when matching a query to
a note (e.g. ``SSRI`` -> ``এসসিটালোপ্রাম``).  Spelling variants of ONE word (মেটফরমিন / মেটফর্মিন,
ডায়াবেটিস / ডায়াবেটিস-এর) are handled separately by the skeleton normaliser in ``rag.py`` and are
not listed here.

Written by the team (LLM-assisted) from general clinical knowledge; it is NOT derived from the
test notes or queries and contains no patient data.
"""

_TABLE = """
diabetes: ডায়াবেটিস | ডায়াবেটিক | ডায়াবেটিস মেলিটাস | টাইপ ২ ডায়াবেটিস | T2DM | DM | diabetes | টাইপ-২ ডায়াবেটিস | টাইপ ২ ডায়াবেটিস মেলিটাস | টাইপ ১ ডায়াবেটিস
hba1c: HbA1c | এইচবিএ১সি | HbA১c | গ্লাইকোসাইলেটেড হিমোগ্লোবিন
metformin: মেটফর্মিন | metformin | মেটফরমিন
insulin: ইনসুলিন | insulin
hypertension: উচ্চ রক্তচাপ | হাইপারটেনশন | hypertension | HTN | রক্তচাপ বেশি
heart_failure: হার্ট ফেইলিওর | হার্ট ফেলিওর | হৃদযন্ত্র বিকল | হৃদপিণ্ডের ব্যর্থতা | heart failure | CHF | কনজেস্টিভ হার্ট ফেইলিওর
loop_diuretic: লুপ ডাইইউরেটিক্স | লুপ ডাইইউরেটিক | ফিউরোসেমাইড | ফুরোসেমাইড | টরসেমাইড | ফুরোসেমাইড | furosemide | লাসিক্স | বুমেটানাইড
kidney_disease: কিডনি রোগ | কিডনির সমস্যা | কিডনি রোগ | দীর্ঘস্থায়ী কিডনি রোগ | CKD | সিকেডি | কিডনি বিকল | কিডনির ক্ষতি | কিডনি আঘাত | মূত্রতন্ত্রের রোগ
ischemic_heart: ইস্কিমিক হৃদরোগ | ইসকিমিক হৃদরোগ | করোনারি হৃদরোগ | করোনারি আর্টারি ডিজিজ | CAD | অ্যানজাইনা | হৃদরোগ | ইস্কিমিক | করোনারি | এনজাইনা
chest_pain: বুকে ব্যথা | বুকের ব্যথা | বুক ব্যথা | বুকে চাপ | chest pain | বুকে চাপ ধরা ব্যথা
ecg: ইসিজি | ECG | EKG | ইলেকট্রোকার্ডিওগ্রাম
echo: ইকো | ইকোকার্ডিওগ্রাম | ইকোকার্ডিওগ্রাফি | echo | echocardiogram
hypercholesterolemia: হাইপারকোলেস্টেরোলেমিয়া | কোলেস্টেরল বেশি | রক্তে চর্বি বেশি | ডিসলিপিডেমিয়া | উচ্চ কোলেস্টেরল | dyslipidemia | হাইপারলিপিডেমিয়া | রক্তে চর্বি
statin: স্ট্যাটিন | অ্যাটরভাস্টাটিন | রসুভাস্টাটিন | সিমভাস্টাটিন | atorvastatin | rosuvastatin
lifestyle: জীবনযাপন পরিবর্তন | জীবনযাত্রার পরিবর্তন | লাইফস্টাইল | জীবনযাপনের পরিবর্তন | খাদ্যাভ্যাস | নিয়মিত হাঁটুন | ওজন কমান | ব্যায়াম
asthma: হাঁপানি | অ্যাজমা | asthma | শিশুদের হাঁপানি
cough_night: রাতে কাশি | রাতের কাশি | রাতে কাশছে
inhaler: ইনহেলার | inhaler | সালবিউটামল | বুডেসোনাইড | স্যালবিউটামল | ফর্মোটেরল | মন্টেলুকাস্ট | নেবুলাইজার
pneumonia: নিউমোনিয়া | pneumonia | ফুসফুসের সংক্রমণ | শ্বাসতন্ত্রের সংক্রমণ | নিউমোনিয়ার
resp_infection: শ্বাসতন্ত্রের সংক্রমণ | শ্বাসনালির সংক্রমণ | ব্রংকাইটিস | ব্রঙ্কাইটিস | ব্রংকিওলাইটিস | ফ্যারিঞ্জাইটিস | রেসপিরেটরি ইনফেকশন
dyspnea: শ্বাসকষ্ট | দম ফুরিয়ে | হাঁপিয়ে | শ্বাস নিতে কষ্ট | শ্বাস কষ্ট | breathlessness | dyspnea
amoxicillin: অ্যামোক্সিসিলিন | amoxicillin | অ্যামক্সিসিলিন | অ্যামোক্সিসিলিন-ক্ল্যাভুলানেট
antibiotic: অ্যান্টিবায়োটিক | antibiotic | অ্যামোক্সিসিলিন | অ্যাজিথ্রোমাইসিন | সেফট্রিয়াক্সোন | সিপ্রোফ্লক্সাসিন | সেফিক্সিম | ডক্সিসাইক্লিন | মেট্রোনিডাজল | সেফুরক্সিম | নাইট্রোফুরান্টয়িন | ক্ল্যারিথ্রোমাইসিন | লেভোফ্লক্সাসিন | সেফালেক্সিন | মেরোপেনেম | অ্যামোক্সিসিলিন-ক্ল্যাভুলানেট | কোট্রাইমোক্সাজল | ফ্লুক্লক্সাসিলিন | জেন্টামাইসিন
cxr: বুকের এক্স-রে | বুকের এক্সরে | চেস্ট এক্স-রে | CXR | বুকের এক্স রে
gout: গাউট | গাউটি আর্থ্রাইটিস | gout | গেঁটে বাত
uric_acid: ইউরিক অ্যাসিড | সিরাম ইউরিক অ্যাসিড | uric acid
dengue: ডেঙ্গু | dengue | ডেঙ্গু জ্বর | ডেঙ্গু হেমোরেজিক জ্বর
ns1: NS1 | এনএস১ | NS১ | এনএস 1
platelet: প্লেটলেট | প্লেটলেট কাউন্ট | platelet | প্লেটিলেট
fever: জ্বর | fever | জ্বরে | জ্বর হচ্ছে
viral_fever: ভাইরাল জ্বর | ভাইরাল ফিভার | ভাইরাল সংক্রমণজনিত জ্বর | ভাইরাল সংক্রমণ | তীব্র ভাইরাল ফিভার | ইনফ্লুয়েঞ্জা | ইনফ্লুয়েঞ্জা-সদৃশ অসুখ | সাধারণ সর্দি-কাশি | কমন কোল্ড | ভাইরাল ফ্যারিঞ্জাইটিস | শিশুর জ্বর ও সর্দি-কাশি
typhoid: টাইফয়েড | টাইফয়েড জ্বর | typhoid | এন্টেরিক ফিভার
prolonged_fever: দীর্ঘদিনের জ্বর | দীর্ঘস্থায়ী জ্বর | অনেকদিন ধরে জ্বর | ৭ দিন ধরে জ্বর | সপ্তাহ ধরে জ্বর
peptic_ulcer: পেপটিক আলসার | পাকস্থলীর আলসার | গ্যাস্ট্রিক আলসার | ডুওডেনাল আলসার | আলসার | peptic ulcer
epigastric_burning: পেটের উপরের অংশে জ্বালাপোড়া | পেটের উপরের অংশে জ্বালা | অম্বল | বুকজ্বালা | পেটে জ্বালাপোড়া | জ্বালাপোড়া | অ্যাসিডিটি | গ্যাস্ট্রিক
ppi: PPI | পিপিআই | প্রোটন পাম্প ইনহিবিটর | ওমিপ্রাজল | এসোমেপ্রাজল | প্যান্টোপ্রাজল | র‍্যাবেপ্রাজল | ল্যানসোপ্রাজল | omeprazole | pantoprazole | esomeprazole | প্যান্টোপ্রাজোল
nsaid: NSAID | NSAIDs | এনএসএআইডি | ব্যথানাশক ওষুধ | ডাইক্লোফেনাক | ইবুপ্রোফেন | নেপ্রক্সেন | ইটোরিকক্সিব | কেটোরোলাক | ইন্ডোমেথাসিন
allergic_rhinitis: অ্যালার্জিক রাইনাইটিস | অ্যালার্জিক নাক | রাইনাইটিস | hay fever | মৌসুমি অ্যালার্জিক রাইনাইটিস | allergic rhinitis
sneezing: হাঁচি | sneezing | নাক চুলকানো | নাক দিয়ে পানি পড়া
antihistamine: অ্যান্টিহিস্টামিন | antihistamine | সিটিরিজিন | লেভোসিটিরিজিন | ফেক্সোফেনাডিন | লোরাটাডিন | ক্লোরফেনিরামিন | cetirizine | fexofenadine | levocetirizine
scabies: স্ক্যাবিস | খোসপাঁচড়া | scabies | খোস
night_itch: রাতে চুলকানি | রাতে চুলকায় | রাতে বেশি চুলকানি | রাতে চুলকানির
permethrin: পারমেথ্রিন | permethrin
depression: বিষণ্নতা | ডিপ্রেশন | depression | মন খারাপ
ssri: SSRI | এসএসআরআই | এসসিটালোপ্রাম | সার্ট্রালিন | ফ্লুক্সেটিন | প্যারোক্সেটিন | সিটালোপ্রাম | escitalopram | sertraline | fluoxetine
fatigue: ক্লান্তি | দুর্বলতা | ক্লান্ত | দুর্বল | অবসাদ | fatigue | weakness
sepsis: সেপসিস | sepsis | রক্তে সংক্রমণ | সেপটিসেমিয়া
confusion: বিভ্রান্তি | কনফিউশন | confusion | বিভ্রান্ত | মানসিক বিভ্রান্তি
blood_test: রক্ত পরীক্ষা | রক্তের পরীক্ষা | blood test | রক্ত কালচার | রক্তের কালচার | সিবিসি | CBC
ct_scan: সিটি স্ক্যান | CT scan | সিটি | CT
urticaria: আর্টিকারিয়া | আমবাত | urticaria | চাকা চাকা দাগ | তীব্র আর্টিকারিয়া
bph: প্রস্টেট বৃদ্ধি | BPH | বিপিএইচ | প্রস্টেটের আকার বর্ধিত | প্রস্টেট বড়
tamsulosin: ট্যামসুলোসিন | টামসুলোসিন | tamsulosin
tb: যক্ষ্মা | টিবি | TB | tuberculosis | ফুসফুসের যক্ষ্মা
chronic_cough: দীর্ঘস্থায়ী কাশি | দীর্ঘদিনের কাশি | অনেকদিন ধরে কাশি | সপ্তাহ ধরে কাশি
weight_loss: ওজন কমা | ওজন কমে | ওজন কমছে | ওজন হ্রাস | weight loss
night_sweat: রাতে ঘাম | রাতে ঘাম দেয় | night sweats | রাতে ঘামে
paracetamol: প্যারাসিটামল | paracetamol | নাপা | acetaminophen
paracetamol_syrup: প্যারাসিটামল সিরাপ
herpes_zoster: হারপিস জোস্টার | হারপিস | শিংলস | herpes zoster | shingles
blister: ফোসকা | ফোস্কা | ফোসকার | blister | ফোসকাযুক্ত
acyclovir: অ্যাসাইক্লোভির | acyclovir | অ্যাসাইক্লোভির
migraine: মাইগ্রেন | migraine
nausea_vomiting: বমি | বমি বমি ভাব | nausea | vomiting
photophobia: আলোতে অস্বস্তি | আলো সহ্য হয় না | photophobia | আলোতে কষ্ট
pcos: PCOS | পিসিওএস | পলিসিস্টিক ওভারি | পলিসিস্টিক ওভারি সিনড্রোম
irregular_periods: মাসিক অনিয়মিত | অনিয়মিত মাসিক | মাসিকের অনিয়ম | মাসিক অনিয়মিত
uti: UTI | ইউটিআই | মূত্রনালির সংক্রমণ | মূত্রনালীর সংক্রমণ | প্রস্রাবের সংক্রমণ | সিস্টাইটিস | তীব্র সিস্টাইটিস | মূত্রথলির সংক্রমণ
dysuria: প্রস্রাবে জ্বালাপোড়া | প্রস্রাবে জ্বালা | প্রস্রাবে ব্যথা | dysuria
urine_re: urine R/E | ইউরিন R/E | ইউরিন আরই | প্রস্রাবের রুটিন | প্রস্রাব পরীক্ষা | প্রস্রাবের রুটিনে | ইউরিন রুটিন | urine RE
diarrhea: পাতলা পায়খানা | ডায়রিয়া | diarrhea | diarrhoea | পাতলা পায়খানা হচ্ছে | গ্যাস্ট্রোএন্টেরাইটিস
dehydration: জল শূন্যতা | পানিশূন্যতা | dehydration | জলশূন্যতা | ডিহাইড্রেশন
ors: ওআরএস | ORS | খাবার স্যালাইন | ওরাল স্যালাইন
"""


def _parse():
    groups = {}
    for line in _TABLE.strip().splitlines():
        if ":" not in line:
            continue
        name, variants = line.split(":", 1)
        groups[name.strip()] = [v.strip() for v in variants.split("|") if v.strip()]
    return groups


CONCEPTS = _parse()
