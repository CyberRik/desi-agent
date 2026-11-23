"""[R] Phrase bank for the round-2 grounding set (scripts/data_round2.py), written once and saved as JSON.

Hindi and Bengali are written here in ITRANS (ASCII) and converted to Devanagari / Bengali script with
`indic_transliteration`, so this source stays ASCII. Only this build step needs that package; data_round2.py reads
the JSON it writes (data/round2/phrases.json). Placeholders `<<name>>` are filled later; `<<=TEXT>>` is literal
Latin text kept as-is (e.g. UPI). Digits and `*` pass through untouched.

    pip install indic_transliteration
    python scripts/data_round2_phrases.py --out data/round2/phrases.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

O = "\u0949"  # Devanagari vowel sign candra o (as in "block"), not in plain ITRANS

# ---------------------------------------------------------------------------------------------------------------
# Hindi (ITRANS)
HI = {
    "opener": ["", "", "namaste", "suniye", "helo"],
    "block": {
        "problem": {
            "lost": ["meraa DebiTa kaarDa kho gayaa hai", "mujhase meraa kaarDa kahIM gira gayaa hai",
                     "kala se meraa kreDiTa kaarDa nahIM mila rahaa, shAyada kho gayaa hai"],
            "stolen": ["meraa baTuA chorI ho gayaa aura usameM meraa kaarDa thaa", "kisI ne meraa kaarDa churA liyA hai",
                       "bAzAra meM meraa kaarDa chorI ho gayaa"],
            "fraud": ["mere kaarDa se kuchha aise pemeMTa hue haiM jo maiMne nahIM kie",
                      "lagatA hai mere kaarDa ke saatha dhokhAdha.DI huI hai"],
            "damaged": ["meraa kaarDa TUTa gayaa hai", "mere kaarDa kI chipa kharAba ho gaI hai"],
        },
        "generic": [f"mujhe apanA kaarDa turaMta bla{O}ka karavAnA hai", f"mere kaarDa ko abhI bla{O}ka kara dIjie"],
        "request": [f"kR^ipayaa ise turaMta bla{O}ka kara dIjie", f"isako jaldI se bla{O}ka kara do",
                    f"kaarDa bla{O}ka karavAnA hai"],
        "detail": ["kaarDa ke AkhirI chAra aMka <<last4>> haiM", "kaarDa nambara <<last4>> para khatma hotA hai"],
    },
    "dispute": {
        "problem": {
            "duplicate": ["eka hI pemeMTa ke lie mere khAte se do bAra paise kaTa gae",
                          "eka hI lena-dena ke lie mujhase do bAra paise kATe gae"],
            "unauthorized": ["mere khAte se eka pemeMTa huA hai jo maiMne kiyA hI nahIM",
                             "mere khAte se kisI ne binA batAe paise nikAle haiM"],
            "wrong_amount": ["mujhase galata rakama kATI gaI hai", "bila se zyAdA paise kaTa gae haiM"],
            "not_received": ["maiMne paise de die lekina sAmAna abhI taka nahIM milA",
                             "pemeMTa ho gayA para sevA mujhe milI hI nahIM"],
        },
        "request": ["kR^ipayaa isa para DispyUTa darja kara dIjie", "isakI shikAyata darja kIjie",
                    "mujhe isa para DispyUTa uThAnA hai"],
        "detail": ["lena-dena kI AIDI <<txn>> hai", "TrAMjaikshana AIDI <<txn>> hai"],
    },
    "upi": {
        "problem": {
            "pending": ["maiMne <<=UPI>> se paise bheje the para pemeMTa abhI taka aTakA huA hai",
                        "meraa <<=UPI>> pemeMTa kAfI dera se aTakA hai"],
            "failed": ["meraa <<=UPI>> pemeMTa phela ho gayA lekina khAte se paise kaTa gae"],
            "not_received": ["jise <<=UPI>> se paise bheje unheM abhI taka nahIM mile"],
        },
        "request": ["kR^ipayaa isakA sTeTasa batAie", "zarA jAMcha karake batAie ki kyA huA"],
        "detail": ["<<=UPI>> rephareMsa naMbara <<ref>> hai"],
    },
    "transfer": {
        "rel": ["apane bhAI ko", "apanI bahana ko", "apanI mammI ko", "pApA ko", "apane dosta ko"],
        "no_upi": ["mujhe <<rel>> <<amt>> rupaye bhejane haiM", "<<rel>> <<amt>> rupaye TrAMsaphara kara dIjie",
                   "kR^ipayaa <<rel>> <<amt>> rupaye bheja do"],
        "no_amount": ["<<upi>> para kuchha paise bhejane haiM", "<<upi>> ko paise TrAMsaphara kara dIjie",
                      "mujhe <<upi>> ko paise bhejane haiM"],
        "control": ["<<upi>> ko <<amt>> rupaye bheja dIjie", "mujhe <<upi>> para <<amt>> rupaye bhejane haiM"],
    },
    "reply": {
        "clarify_block": [f"maiM ApakA kaarDa bla{O}ka karane meM madada kara sakatA hU.N | kR^ipayaa kaarDa ke AkhirI 4 aMka batAie |"],
        "clarify_block_reason": [f"maiM ApakA kaarDa bla{O}ka karane meM madada kara sakatA hU.N | kR^ipayaa kaarDa ke AkhirI 4 aMka "
                                 "batAie aura kAraNa bhI batAie: kho gayA, chorI, dhokhAdha.DI yA kharAba |"],
        "clarify_dispute": ["maiM Apake lie DispyUTa darja kara sakatA hU.N | kR^ipayaa lena-dena kI AIDI batAie, yA kaheM to "
                            "maiM ApakI hAla kI lena-dena kI sUchI dikhA sakatA hU.N |"],
        "clarify_upi": ["maiM isakI jAMcha kara sakatA hU.N | kR^ipayaa pemeMTa kA 12 aMkoM kA <<=UPI>> rephareMsa naMbara batAie |"],
        "clarify_transfer_upi": ["zarUra | kR^ipayaa jinheM paise bhejane haiM unakI <<=UPI>> AIDI batAie |"],
        "clarify_transfer_amount": ["zarUra | kR^ipayaa batAie ki <<upi>> ko kitane rupaye bhejane haiM?"],
        "block": [f"ApakA kaarDa ****<<last4>> bla{O}ka kara diyA gayA hai | rephareMsa: <<ref>> | nayA kaarDa <<eta>> dina meM "
                  "Apake pAsa A jAegA |"],
        "dispute": ["maiMne <<txn>> para DispyUTa darja kara diyA hai | ApakI DispyUTa AIDI <<dsp>> hai, isakA samAdhAna "
                    "<<days>> dina meM hogA |"],
        "upi_success": ["ApakA <<=UPI>> pemeMTa <<ref>> saphala rahA hai |"],
        "upi_pending": ["ApakA <<=UPI>> pemeMTa <<ref>> abhI prakriyA meM hai | kR^ipayaa kuchha dera bAda phira dekheM |"],
        "upi_failed": ["ApakA <<=UPI>> pemeMTa <<ref>> phela ho gayA hai | rakama 48 ghaMToM meM Apake khAte meM vApasa A "
                       "jAegI |"],
        "transfer": ["ho gayA | <<amt>> rupaye <<upi>> ko bheje jA chuke haiM | ApakA <<=UPI>> rephareMsa: <<ref>> |"],
    },
}

# ---------------------------------------------------------------------------------------------------------------
# Bengali (ITRANS)
BN = {
    "opener": ["", "", "namaskAra", "shununa", "hyAlo"],
    "block": {
        "problem": {
            "lost": ["aamaara DebiTa kaarDa hAriYe gechhe", "aamaara kaarDaTA kothAo hAriYe phelechhi"],
            "stolen": ["aamaara mAnibyAga churi hoYe gechhe, tAte kaarDa chhila", "kAla bAse aamaara kaarDa churi hoYe gechhe"],
            "fraud": ["aamaara kaarDa theke emana kichhu pemenTa hoYechhe yA aami karini",
                      "mane hachchhe aamaara kaarDe jAliYAti hoYechhe"],
            "damaged": ["aamaara kaarDaTA bhe~Nge gechhe", "aamaara kaarDera chipa khArApa hoYe gechhe"],
        },
        "generic": ["aamaara kaarDaTA ekhuni blaka karAte chAi", "aamaara kaarDa ekhuni blaka kare dina"],
        "request": ["daYA kare ekhuni blaka kare dina", "tA.DAtA.Di kaarDaTA blaka karuna", "kaarDaTA blaka karAte chAi"],
        "detail": ["kaarDera shesha chAra a~Nka <<last4>>", "kaarDa nambarera shesha chAraTi saMkhyA <<last4>>"],
    },
    "dispute": {
        "problem": {
            "duplicate": ["ekaTA pemenTera janya aamaara khAtA theke dubAra TAkA keTe niYechhe"],
            "unauthorized": ["aamaara khAtA theke ekaTA pemenTa hoYechhe yA aami karini",
                             "aamaara anumati chhA.DAi khAtA theke TAkA keTechhe"],
            "wrong_amount": ["aamaara kAchha theke bhula TAkA keTechhe, bilera cheYe beshi"],
            "not_received": ["aami TAkA diYechhi kintu jinisa ekhano pAini", "pemenTa hoYe gechhe kintu parishebA pAini"],
        },
        "request": ["daYA kare ekaTA DisapiuTa tuluna", "ei niYe abhiyoga jAnAte chAi", "eTAra upara DisapiuTa tulate chAi"],
        "detail": ["lenadenera AiDi <<txn>>", "TrAnajyAkashana AiDi <<txn>>"],
    },
    "upi": {
        "problem": {
            "pending": ["aami <<=UPI>> diYe TAkA pAThiYechhi kintu pemenTaTA ekhano ATake Achhe",
                        "aamaara <<=UPI>> pemenTa anekShaNa dhare ATake Achhe"],
            "failed": ["aamaara <<=UPI>> pemenTa pheila hoYechhe kintu TAkA keTe gechhe"],
            "not_received": ["yAke <<=UPI>> diYe TAkA pAThiYechhi se ekhano pAYani"],
        },
        "request": ["daYA kare pemenTera abasthA jAnAna", "ekaTu dekhe baluna ki hoYechhe"],
        "detail": ["<<=UPI>> rephArensa nambara <<ref>>"],
    },
    "transfer": {
        "rel": ["aamaara bhAike", "aamaara bonake", "mAke", "bAbAke", "aamaara bandhuke"],
        "no_upi": ["<<rel>> <<amt>> TAkA pAThAte chAi", "daYA kare <<rel>> <<amt>> TAkA pAThiYe dina",
                   "<<rel>> <<amt>> TAkA TrAnsphAra kare dina"],
        "no_amount": ["<<upi>>-te kichhu TAkA pAThAte chAi", "<<upi>>-ke TAkA TrAnsphAra kare dina"],
        "control": ["<<upi>>-ke <<amt>> TAkA pAThiYe dina", "<<upi>>-te <<amt>> TAkA pAThAte chAi"],
    },
    "reply": {
        "clarify_block": ["aami ApanAra kaarDa blaka karate sAhAyya karate pAri | daYA kare kaarDera shesha 4Ti a~Nka jAnAna |"],
        "clarify_block_reason": ["aami ApanAra kaarDa blaka karate sAhAyya karate pAri | daYA kare kaarDera shesha 4Ti a~Nka "
                                 "jAnAna, Ara kAraNaTAo baluna: hAriYechhe, churi, jAliYAti naki kharApa |"],
        "clarify_dispute": ["aami ApanAra janya DisapiuTa tulate pAri | daYA kare lenadenera AiDi jAnAna, athabA balale aami "
                            "ApanAra sAmpratika lenadena dekhe nite pAri |"],
        "clarify_upi": ["aami eTA dekhe dite pAri | daYA kare pemenTera 12 a~Nkera <<=UPI>> rephArensa nambara jAnAna |"],
        "clarify_transfer_upi": ["nishchaYa | daYA kare yAke TAkA pAThAte chAna tA.Nra <<=UPI>> AiDi jAnAna |"],
        "clarify_transfer_amount": ["nishchaYa | daYA kare jAnAna <<upi>>-ke kata TAkA pAThAte chAna?"],
        "block": ["ApanAra kaarDa ****<<last4>> blaka karA hoYechhe | rephArensa: <<ref>> | natuna kaarDa <<eta>> dinera madhye "
                  "ApanAra kAchhe pau.Nchhe yAbe |"],
        "dispute": ["aami <<txn>>-era upara DisapiuTa tule diYechhi | ApanAra DisapiuTa AiDi <<dsp>>, <<days>> dinera madhye "
                    "samAdhAna habe |"],
        "upi_success": ["ApanAra <<=UPI>> pemenTa <<ref>> saphala hoYechhe |"],
        "upi_pending": ["ApanAra <<=UPI>> pemenTa <<ref>> ekhano prakriYAdhIna | kichhukShaNa pare Abara dekhuna |"],
        "upi_failed": ["ApanAra <<=UPI>> pemenTa <<ref>> pheila hoYechhe | TAkA 48 ghaNTAra madhye ApanAra khAtAYa pherata "
                       "Asabe |"],
        "transfer": ["hoYe gechhe | <<upi>>-ke <<amt>> TAkA pAThAno hoYechhe | ApanAra <<=UPI>> rephArensa: <<ref>> |"],
    },
}

# ---------------------------------------------------------------------------------------------------------------
# Hinglish / Banglish / English (Latin, written directly)
HI_EN = {
    "opener": ["", "", "Hello", "Namaste", "Bhai"],
    "block": {
        "problem": {
            "lost": ["Mera debit card kho gaya hai", "Mujhse mera card kahin gir gaya hai",
                     "Kal se mera credit card nahi mil raha, shayad kho gaya hai"],
            "stolen": ["Mera wallet chori ho gaya aur usme mera card tha", "Kisi ne mera card chura liya hai"],
            "fraud": ["Mere card se kuch aise payments hue hain jo maine nahi kiye", "Lagta hai mere card ke saath fraud hua hai"],
            "damaged": ["Mera card toot gaya hai", "Mere card ki chip kharab ho gayi hai"],
        },
        "generic": ["Mujhe apna card abhi turant block karwana hai", "Mera card abhi ke abhi block kar dijiye"],
        "request": ["please isko turant block kar do", "jaldi se block kar dijiye", "card block karwana hai"],
        "detail": ["card ke last 4 digits <<last4>> hain", "card number <<last4>> pe khatam hota hai"],
    },
    "dispute": {
        "problem": {
            "duplicate": ["Ek hi payment ke liye mere account se do baar paise kat gaye"],
            "unauthorized": ["Mere account se ek payment hua hai jo maine kiya hi nahi",
                             "Mere account se bina bataye kisi ne paise nikale hain"],
            "wrong_amount": ["Mujhse galat amount kata hai, bill se zyada", "Bill se zyada paise kat gaye hain"],
            "not_received": ["Maine payment kar diya lekin saaman abhi tak nahi mila", "Payment ho gaya par service mili hi nahi"],
        },
        "request": ["please iska dispute raise kar do", "is par complaint darj kijiye", "mujhe iska dispute karna hai"],
        "detail": ["transaction ID <<txn>> hai", "txn id hai <<txn>>"],
    },
    "upi": {
        "problem": {
            "pending": ["Maine UPI se paise bheje the par payment abhi tak pending dikha raha hai",
                        "Mera UPI payment kaafi der se atka hua hai"],
            "failed": ["Mera UPI payment fail ho gaya lekin account se paise kat gaye"],
            "not_received": ["Jisko UPI se paise bheje usko abhi tak nahi mile"],
        },
        "request": ["please status check karke batao", "zara check kijiye kya hua"],
        "detail": ["UPI ref number <<ref>> hai", "reference number hai <<ref>>"],
    },
    "transfer": {
        "rel": ["apne bhai ko", "apni behen ko", "mummy ko", "papa ko", "apne dost ko"],
        "no_upi": ["Mujhe <<rel>> <<amt>> rupaye bhejne hain", "<<rel>> <<amt>> rupaye transfer kar dijiye",
                   "Please <<rel>> <<amt>> rupaye bhej do"],
        "no_amount": ["<<upi>> pe kuch paise bhejne hain", "<<upi>> ko paise transfer kar dijiye", "Mujhe <<upi>> ko paise bhejne hain"],
        "control": ["<<upi>> ko <<amt>> rupaye bhej dijiye", "Mujhe <<upi>> pe <<amt>> rupaye bhejne hain"],
    },
    "reply": {
        "clarify_block": ["Main aapka card block karne mein madad kar sakta hoon. Please card ke last 4 digits bataiye."],
        "clarify_block_reason": ["Main aapka card block karne mein madad kar sakta hoon. Please card ke last 4 digits bataiye, aur "
                                 "reason bhi bataiye: kho gaya, chori, fraud ya kharab."],
        "clarify_dispute": ["Main aapke liye dispute raise kar sakta hoon. Please transaction ID bataiye, ya kahen to main aapke "
                            "recent transactions check karke dikha deta hoon."],
        "clarify_upi": ["Main iska status check kar sakta hoon. Please payment ka 12 digit UPI reference number bataiye."],
        "clarify_transfer_upi": ["Zaroor. Please jinko paise bhejne hain unki UPI ID bataiye."],
        "clarify_transfer_amount": ["Zaroor. Please bataiye <<upi>> ko kitne rupaye bhejne hain?"],
        "block": ["Aapka card ****<<last4>> block kar diya gaya hai. Reference: <<ref>>. Naya card <<eta>> din mein aapke paas aa "
                  "jayega."],
        "dispute": ["Maine <<txn>> par dispute raise kar diya hai. Aapki dispute ID <<dsp>> hai, <<days>> din mein iska samadhan "
                    "ho jayega."],
        "upi_success": ["Aapka UPI payment <<ref>> successful raha hai."],
        "upi_pending": ["Aapka UPI payment <<ref>> abhi pending hai. Thodi der baad phir check kijiye."],
        "upi_failed": ["Aapka UPI payment <<ref>> fail ho gaya hai. Paise 48 ghante mein aapke account mein wapas aa jayenge."],
        "transfer": ["Ho gaya. <<amt>> rupaye <<upi>> ko bhej diye gaye hain. Aapka UPI reference: <<ref>>."],
    },
}

BN_EN = {
    "opener": ["", "", "Hello", "Nomoskar", "Dada"],
    "block": {
        "problem": {
            "lost": ["Amar debit card ta hariye geche", "Amar card ta kothao hariye felechi"],
            "stolen": ["Amar wallet churi hoye geche, tate card chilo", "Kal bus e amar card churi hoye geche"],
            "fraud": ["Amar card theke emon kichu payment hoyeche ja ami korini", "Mone hocche amar card e fraud hoyeche"],
            "damaged": ["Amar card ta bhenge geche", "Amar card er chip kharap hoye geche"],
        },
        "generic": ["Amar card ta ekhuni block korte chai", "Amar card ta ekhuni block kore din"],
        "request": ["please ekhuni block kore dao", "taratari card ta block korun", "card ta block korte chai"],
        "detail": ["card er last 4 digit <<last4>>", "card er shesh char ta number <<last4>>"],
    },
    "dispute": {
        "problem": {
            "duplicate": ["Ekta payment er jonno amar account theke duibar taka kete niyeche"],
            "unauthorized": ["Amar account theke ekta payment hoyeche ja ami korini",
                             "Amar permission chara account theke taka keteche"],
            "wrong_amount": ["Amar theke bhul amount kete niyeche, bill er cheye beshi"],
            "not_received": ["Ami taka diyechi kintu jinish ekhono pai ni", "Payment hoye geche kintu service pai ni"],
        },
        "request": ["please eta niye dispute raise kore din", "ei niye complaint korte chai", "eta te dispute tulun"],
        "detail": ["transaction ID holo <<txn>>", "txn id <<txn>>"],
    },
    "upi": {
        "problem": {
            "pending": ["Ami UPI diye taka pathiyechi kintu payment ta ekhono atke ache",
                        "Amar UPI payment onekkhon dhore atke ache"],
            "failed": ["Amar UPI payment fail hoyeche kintu taka kete geche"],
            "not_received": ["Jake UPI diye taka pathiyechi se ekhono pay ni"],
        },
        "request": ["please status ta check kore bolun", "ektu dekhe bolo ki hoyeche"],
        "detail": ["UPI ref number <<ref>>", "reference number holo <<ref>>"],
    },
    "transfer": {
        "rel": ["amar bhai ke", "amar bon ke", "maa ke", "baba ke", "amar bondhu ke"],
        "no_upi": ["<<rel>> <<amt>> taka pathate chai", "Please <<rel>> <<amt>> taka pathiye din",
                   "<<rel>> <<amt>> taka transfer kore dao"],
        "no_amount": ["<<upi>> e kichu taka pathate chai", "<<upi>> ke taka transfer kore din", "Ami <<upi>> ke taka pathate chai"],
        "control": ["<<upi>> ke <<amt>> taka pathiye din", "Ami <<upi>> e <<amt>> taka pathate chai"],
    },
    "reply": {
        "clarify_block": ["Ami apnar card block korte sahajjo korte pari. Please card er shesh 4 ta digit bolun."],
        "clarify_block_reason": ["Ami apnar card block korte sahajjo korte pari. Please card er shesh 4 ta digit bolun, ar karon "
                                 "ta o janan: hariyeche, churi, fraud naki kharap."],
        "clarify_dispute": ["Ami apnar jonno dispute raise korte pari. Please transaction ID ta bolun, othoba bolle ami apnar recent "
                            "transaction gulo dekhe nite pari."],
        "clarify_upi": ["Ami eta check kore dite pari. Please payment er 12 digit er UPI reference number ta bolun."],
        "clarify_transfer_upi": ["Nishchoi. Please jake taka pathate chan tar UPI ID ta bolun."],
        "clarify_transfer_amount": ["Nishchoi. Bolun <<upi>> ke koto taka pathate chan?"],
        "block": ["Apnar card ****<<last4>> block kora hoyeche. Reference: <<ref>>. Notun card <<eta>> diner moddhe apnar kache "
                  "pouche jabe."],
        "dispute": ["Ami <<txn>> er upor dispute raise kore diyechi. Apnar dispute ID <<dsp>>, <<days>> diner moddhe somadhan "
                    "hobe."],
        "upi_success": ["Apnar UPI payment <<ref>> successful hoyeche."],
        "upi_pending": ["Apnar UPI payment <<ref>> ekhono pending ache. Kichukkhon pore abar check korun."],
        "upi_failed": ["Apnar UPI payment <<ref>> fail hoyeche. Taka 48 ghontar moddhe apnar account e ferot asbe."],
        "transfer": ["Hoye geche. <<upi>> ke <<amt>> taka pathano hoyeche. Apnar UPI reference: <<ref>>."],
    },
}

EN = {
    "opener": ["", "", "Hi", "Hello", "Good morning"],
    "block": {"detail": ["The card ends in <<last4>>.", "My card number ends with <<last4>>.",
                         "It's the card ending <<last4>>."]},
    "dispute": {"detail": ["Please raise a dispute on transaction <<txn>>.", "The transaction ID is <<txn>>, please dispute it.",
                           "Can you open a dispute for <<txn>>?"]},
    "upi": {"detail": ["The UPI reference number is <<ref>>.", "UPI ref: <<ref>>. Can you check its status?"]},
    "transfer": {
        "rel": ["my brother", "my sister", "my mom", "my dad", "a friend"],
        "no_upi": ["I need to send Rs <<amt>> to <<rel>>", "Please transfer <<amt>> rupees to <<rel>>",
                   "Send Rs <<amt>> to <<rel>> right now"],
        "no_amount": ["Please send some money to <<upi>>", "I want to transfer money to <<upi>>", "Pay <<upi>> for me please"],
        "control": ["Send Rs <<amt>> to <<upi>> please", "I want to transfer <<amt>> rupees to <<upi>>"],
    },
    "reply": {
        "clarify_block": ["I can help you block your card. Could you please share the last 4 digits of the card?"],
        "clarify_block_reason": ["I can help you block your card. Could you please share the last 4 digits of the card and tell "
                                 "me the reason: lost, stolen, fraud or damaged?"],
        "clarify_dispute": ["I can raise a dispute for you. Could you please share the transaction ID? If you don't have it, I "
                            "can look up your recent transactions so you can tell me which one it is."],
        "clarify_upi": ["I can check that for you. Could you please share the 12-digit UPI reference number of the payment?"],
        "clarify_transfer_upi": ["Sure. Could you please share the UPI ID of the person you want to send the money to?"],
        "clarify_transfer_amount": ["Sure. How much money would you like to send to <<upi>>?"],
        "block": ["Your card ****<<last4>> has been blocked. Reference: <<ref>>. A replacement card will reach you in <<eta>> "
                  "days."],
        "dispute": ["I have raised a dispute on <<txn>>. Your dispute ID is <<dsp>>, and it should be resolved within <<days>> "
                    "days."],
        "upi_success": ["Your UPI payment <<ref>> was successful."],
        "upi_pending": ["Your UPI payment <<ref>> is still pending. Please check again after some time."],
        "upi_failed": ["Your UPI payment <<ref>> failed. The amount will be auto-reversed to your account within 48 hours."],
        "transfer": ["Done. Rs <<amt>> has been sent to <<upi>>. Your UPI reference: <<ref>>."],
        "no_tool_cancel": ["I'm sorry, I can't cancel or reverse a transfer from this chat. Please call customer care as soon "
                           "as possible. If you want, I can check the status of a UPI payment if you share its reference "
                           "number."],
    },
}

KEEP = re.compile(r"(<<[^>]*>>|[0-9]+|\*+|\?)")


def translit(text: str, script) -> str:
    from indic_transliteration import sanscript
    out = []
    for part in KEEP.split(text):
        if not part:
            continue
        if KEEP.fullmatch(part):
            out.append(part[3:-2] if part.startswith("<<=") else part)
        else:
            out.append(sanscript.transliterate(part, sanscript.ITRANS, script))
    s = "".join(out).replace(" \u0964", "\u0964").strip()
    leftover = re.sub(r"<<[^>]*>>|UPI", "", s)
    assert not re.search(r"[A-Za-z]", leftover), f"untransliterated Latin left in: {ascii(s)}"
    return s


def convert(tree, script):
    if isinstance(tree, dict):
        return {k: convert(v, script) for k, v in tree.items()}
    if isinstance(tree, list):
        return [convert(v, script) for v in tree]
    return translit(tree, script) if tree else tree


def main():
    from indic_transliteration import sanscript
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/round2/phrases.json")
    a = p.parse_args()
    bank = {"en": EN, "hi": convert(HI, sanscript.DEVANAGARI), "bn": convert(BN, sanscript.BENGALI), "hi-en": HI_EN,
            "bn-en": BN_EN}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"[R] wrote {a.out}")


if __name__ == "__main__":
    main()
