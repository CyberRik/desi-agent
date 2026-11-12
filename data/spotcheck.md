# Test-set spot check (R4)

Scores only use the tool calls, so judge the CUSTOMER message and the CALL lines: does the message make sense, and are the calls what a good bank agent should do? Bad BOT wording matters less (it is not scored).
To reject an episode, add its id to `data/spotcheck_rejected.txt` (one per line). Rejected ids are removed from the test set without changing the others.

## 1. Flagged by the LLM judge (41): please decide each one

**bn-clarify-dispute_no_txn-01659** `incomplete`: বট বাকি তথ্য চায় কিন্তু সম্পূর্ণ নয়

- USER: আমার অ্যাকাউন্টে একটা ভুল চার্জ হয়েছে কিন্তু আমি কীভাবে এটা ডিসপিউট করব জানি না, দয়া করে সাহায্য করুন।
- BOT: আপনি কোন ট্রানজেকশন আইডি দিয়ে চার্জটি করা হয়েছে তা জানান কিন্তু আপনি কীভাবে এটা ডিসপিউট করবেন জানি না, দয়া করে সাহায্য করুন।

**bn-clarify-transfer_no_amount-01634** `incomplete`: বট প্রয়োজনীয় তথ্য জিজ্ঞাসা করে না

- USER: আমি আমার ভাইকে টাকা পাঠাতে চাই। কীভাবে করব?
- BOT: আপনি আপনার ভাইয়ের UPI আইডি এবং পাঠাতে চান টাকা পরিমাণ দিন।

**bn-multi_tool-block_then_balance-01580** `incomplete`: বাকি পরিমাণ জানান হয়নি

- USER: আমার ক্রেডিট কার্ড 6296 চুরি হয়ে গেল, তা ব্লক করুন এবং আমার বর্তমান ক্রেডিট কার্ডের বাকি পরিমাণ জানান।
- CALL: `block_card({"card_last4": "6296", "reason": "stolen"})`
- RESULT: `{"card": "****6296", "reason": "stolen", "reference": "BLK63940643", "replacement_eta_days": 5, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: আপনার ক্রেডিট কার্ড ****6296 চুরির জন্য ব্লক করা হয়েছে, রেফারেন্স নম্বর BLK63940643। আপনার কার্ডে বর্তমানে ₹87,159.50 ব্যবহারযোগ্য সীমা রয়েছে।

**bn-no_tool-no_tool-01728** `incomplete`: ব্যাংকিং বিষয়ে তথ্য দেওয়া হয়েছে, কিন্তু রেসিপি সম্পর্কে কিছু নয়

- USER: আমার একটি ভালো বিরিয়ানি রেসিপি দরকার, কিন্তু আমার মোবাইল ব্যাংকিংয়ের মাধ্যমে কীভাবে কিছু পাওয়া যায় তা আমি জানি না।
- BOT: আপনি মোবাইল ব্যাংকিংয়ের মাধ্যমে আপনার অ্যাকাউন্টের ব্যালেন্স, ট্রানজেকশন, কার্ড, UPI, ট্রান্সফার, ATM এবং EMI সম্পর্কে তথ্য পেতে পারবেন। আপনি আমাদের অফিসিয়াল অ্যাপ বা ওয়েবসাইটে লগইন করুন।

**bn-no_tool-no_tool-01751** `incomplete`: বট সংক্ষিপ্ত উত্তর দিতে ব্যর্থ হয়েছে

- USER: আপনি কি একটা জোক বলতে পারবেন? আমার মন খারাপ আছে, তাই কিছু হাস্যকর কথা শুনতে চাই।
- BOT: আপনি কিছু হাস্যকর কথা শুনতে চান? তবে আমি ব্যাংকিং সংক্রান্ত বিষয়ে সাহায্য করতে পারি। আমি জোক বলতে পারি, কিন্তু আপনি যদি ব্যাংকিং-সংক্রান্ত প্রশ্নের জন্য আমাকে খুঁজছেন, তবে আমি সেটা করতে পারব।

**bn-no_tool-no_tool-01731** `incomplete`: বট সংক্ষিপ্ত উত্তর দিতে ব্যর্থ হয়েছে

- USER: আমার কাছে এখন একটা ভালো বিরিয়ানি রেসিপি দরকার কারণ আমার স্বামী আজ এটা খেতে চায়, কিন্তু আমি কোনও ভালো রেসিপি জানি না।
- BOT: আপনি কোনও বিরিয়ানি রেসিপি চাইলে, আমরা ব্যাঙ্কিং সেবা সম্পর্কে সাহায্য করতে পারি। আপনার জন্য এটা নয়।

**bn-no_tool-no_tool-01732** `incomplete`: বট সংক্ষিপ্ত উত্তর দিতে ব্যর্থ হয়েছে

- USER: আমার বিয়ারি রেসিপি চাই কিছুটা বাংলায় আর একটু বেশি সহজে বুঝানো হোক। UPI-এর ব্যবহারের চেয়ে বিয়ারি বানানোটা আমার জন্য অনেক কঠিন!
- BOT: আপনি যদি বিয়ারি রেসিপি চান, তবে আমি সাহায্য করতে পারব না, কারণ আমি কেবলমাত্র ব্যাঙ্কিং-সংক্রান্ত প্রশ্নের জন্যই সাহায্য করতে পারি। UPI, ATM, EMI, PIN এগুলোর বিষয়ে আমি সাহায্য করতে পারি।

**bn-single_tool-get_recent_transactions-01235** `incomplete`: reply lacks some transaction details

- USER: আমার শেষ ছয়টি লেনদেন দেখতে চাই। কীভাবে দেখতে পাবো তা জানান।
- CALL: `get_recent_transactions({"count": 6, "txn_type": "all"})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- BOT: আপনার শেষ ছয়টি লেনদেন দেখানো হয়েছে। যেমন, 12 নভেম্বর 2026-এ বিগবাস্কেটের কাছে ₹110.78 ডেবিট হয়েছিল (ট্রানজ্যাকশন আইডি TXN806078)। অন্যদিকে, 11 নভেম্বর 2026-এ অ্যামাজনের রিফান্ডের জন্য ₹47035.88 ক্রেডিট হয়েছিল (ট্রানজ্যাকশন আইডি TXN703588)।

**bn-single_tool-get_recent_transactions-01234** `incomplete`: reply lacks full transaction list

- USER: আমি আমার শেষ 10টি ডেবিট বা খরচের তালিকা দেখতে চাই।
- CALL: `get_recent_transactions({"count": 10, "txn_type": "debit"})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 1749.07, "date": "2026-11-10", "description":`
- BOT: আপনার শেষ 10টি ডেবিট লেনদেনের তালিকা নিম্নরূপ: উদাহরণস্বরূপ, ₹110.78 (12 নভেম্বর 2026, BigBasket, TXN806078), ₹1749.07 (10 নভেম্বর 2026, Airtel recharge, TXN569907) ইত্যাদি। পুরো তালিকা দেখতে আপনি আমাদের অ্যাপ বা ওয়েবসাইটে লগইন করুন।

**bn-en-clarify-upi_no_ref-02853** `incomplete`: Bot didn't ask for missing info

- USER: Amar UPI transaction ta stuck hyeche, status ta kichu bol dao please.
- BOT: Apni UPI transaction ta stuck hyeche keno? Apni 12 digit UPI reference number ta deoa jete pare?

**bn-en-clarify-transfer_no_amount-02816** `incomplete`: Bot didn't ask for missing info

- USER: Amar bhai ke porichoy kore money send korte chai, kivabe korte parbo?
- BOT: Apni bhai ke porichoy kore money send korte chai, tobe apni bhai'r UPI id ta ba amount ta ber kore dao.

**bn-en-clarify-transfer_no_amount-02820** `incomplete`: Bot didn't ask for missing info

- USER: Bhaiya, amar ekta relative ke kichu pene lagbe, ekhane kivabe send korte pari?
- BOT: Apni kichu boliye, payee'r UPI id ta kichu boliye, amount ta kichu boliye. Apni kichu boliye.

**bn-en-clarify-emi_no_rate-02866** `incomplete`: Bot didn't ask for missing info clearly

- USER: EMI koto hobe, ta calculate kore dao please. Rate and tenure ta kichu boli na, ar por por kichu kora jabe?
- BOT: Apni EMI calculate korte chai, ta korte parbo. Apni interest rate ar loan tenure ta boli do. Tumi kichu boli na, ar por por kichu kora jabe?

**bn-en-clarify-dispute_no_txn-02837** `incomplete`: Bot didn't ask for missing info properly

- USER: Amar account e kuchh wrong charge ase, ekhuni check kore dao please.
- BOT: Apni account e kichhu wrong charge ase kintu transaction id ta deoa jabe kina? Apni check kore dao please.

**bn-en-single_tool-get_recent_transactions-02408** `wrong_fact, incomplete`: total amount is incorrect and missing details

- USER: Ami last 6 ta debits dekhi, kichu money koto pora hoi sei info chahi.
- CALL: `get_recent_transactions({"count": 6, "txn_type": "debit"})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 1749.07, "date": "2026-11-10", "description":`
- BOT: Apni last 6 ta debits dekhi, kichu info deoa hoyeche. Last 6 ta debits er moddhe total ₹12,321.33 pora hoi. Reference number TXN806078, TXN569907, TXN145355, TXN748242, TXN370448, TXN791713.

**hi-clarify-dispute_no_txn-01073** `incomplete`: बॉट ने आवश्यक जानकारी के लिए पूछा नहीं

- USER: मैं एक गलत शुल्क के बारे में विवाद करना चाहता हूं, लेकिन मुझे ट्रांज़ैक्शन आईडी नहीं है।
- BOT: आपके पास ट्रांज़ैक्शन आईडी है या आप हालिया लेनदेनों की जांच करना चाहते हैं?

**hi-clarify-block_card_no_card-01024** `incomplete`: बॉट ने आवश्यक जानकारी नहीं पूछी

- USER: मेरे कार्ड को तुरंत ब्लॉक कर दें, कृपया मेरे खाते से अनाधिकृत उपयोग से रोकें।
- BOT: आपका कार्ड ब्लॉक करने के लिए कृपया कार्ड के अंतिम 4 अंक और कार्ड खो गया, चोरी हो गया, धोखाबाजी या क्षतिग्रस्त है इनमें से कारण बताएं।

**hi-clarify-emi_no_rate-01110** `incomplete`: बॉट ने आवश्यक जानकारी नहीं पूछी

- USER: मैं एक ऋण के लिए EMI की गणना करवाना चाहता हूं। कृपया मुझे बताएं कि इसके लिए क्या जानकारी आवश्यक है।
- BOT: आपको ऋण के लिए EMI की गणना करने के लिए ब्याज दर और ऋण की अवधि की जानकारी देनी होगी। कृपया इन दोनों जानकारियों के साथ हमें संपर्क करें।

**hi-clarify-dispute_no_txn-01061** `incomplete`: बॉट ने आवश्यक जानकारी के लिए पूछा नहीं

- USER: मैं एक गलत शुल्क के विवाद में हूं, पर मुझे ट्रांजेक्शन आईडी नहीं मिली है।
- BOT: आपको ट्रांजेक्शन आईडी मिल गई है क्या? या फिर हालिया ट्रांजेक्शन ढूंढें?

**hi-clarify-block_card_no_card-01033** `incomplete`: बॉट ने आवश्यक जानकारी नहीं पूछी

- USER: मेरी कार्ड तुरंत ब्लॉक करवानी है, जल्दी करें।
- BOT: आपकी कार्ड को ब्लॉक करने के लिए, कृपया अंतिम 4 अंक और कार्ड खो गई है, चोरी हो गई है, धोखा दिया गया है या क्षतिग्रस्त है, इनमें से कारण बताएं।

**hi-multi_tool-block_then_balance-00997** `incomplete`: वर्तमान बैलेंस के बारे में जानकारी छोड़ दी गई

- USER: मेरी क्रेडिट कार्ड 9719 चोर ले गए हैं, इसे ब्लॉक करें और मेरा वर्तमान क्रेडिट कार्ड बैलेंस क्या है?
- CALL: `block_card({"card_last4": "9719", "reason": "stolen"})`
- RESULT: `{"card": "****9719", "reason": "stolen", "reference": "BLK77213889", "replacement_eta_days": 7, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: आपकी क्रेडिट कार्ड ****9719 को चोरी के कारण ब्लॉक कर दिया गया है, रेफरेंस नंबर BLK77213889 है। आपके वर्तमान क्रेडिट कार्ड अकाउंट में ₹87,159.50 उपलब्ध सीमा है।

**hi-multi_tool-block_then_balance-00998** `incomplete`: बॉट ने बकाया राशि के बारे में जानकारी नहीं दी।

- USER: मेरा क्रेडिट कार्ड जिसका अंत 4855 है, मैंने खो दिया है, इसे ब्लॉक करें और मेरे क्रेडिट कार्ड में अबतक की बची हुई राशि के बारे में जानकारी दें।
- CALL: `block_card({"card_last4": "4855", "reason": "lost"})`
- RESULT: `{"card": "****4855", "reason": "lost", "reference": "BLK30556739", "replacement_eta_days": 5, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: आपका क्रेडिट कार्ड ****4855 खो जाने के कारण ब्लॉक कर दिया गया है, रेफरेंस नंबर BLK30556739 है। आपके क्रेडिट कार्ड में ₹87,159.50 उपलब्ध लिमिट है।

**hi-no_tool-no_tool-01143** `incomplete`: बॉट ने रेसिपी के बारे में जानकारी नहीं दी।

- USER: मैं एक अच्छा बर्यानी का रेसिपी चाहत हूं। कृपया मुझे बताएं।
- BOT: आपका बर्यानी के लिए एक स्वादिष्ट रेसिपी देने के लिए हम आपकी सहायता कर सकते हैं। हम आपको बताते हैं कि हम बैंकिंग सेवाओं में सहायता कर सकते हैं, जैसे बैलेंस, लेन-देन, कार्ड, UPI, ट्रांसफर, शाखा/ATM, EMI आदि। हम अन्य प्रश्नों में सहायता करने में सक्षम नहीं हैं।

**hi-single_tool-get_recent_transactions-00650** `wrong_fact, incomplete`: लेनदेन विवरण अपूर्ण और त्रुटिपूर्ण हैं

- USER: मैं अपने अंतिम 5 लेनदेन देखना चाहता हूं, कैसे कर सकता हूं?
- CALL: `get_recent_transactions({"count": 5, "txn_type": "all"})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- BOT: आपके अंतिम 5 लेनदेन इस प्रकार हैं: 
11 नवंबर 2026 को ₹47035.88 के क्रेडिट (अमेज़ॉन के रिफंड) और 12 नवंबर 2026 को ₹110.78 के डेबिट (बिगबास्केट) जैसे लेनदेन हैं। विस्तार से जानने के लिए आपके एपीआई में देख सकते हैं।

**hi-en-clarify-dispute_no_txn-02242** `incomplete`: Bot didn't ask for missing info directly

- USER: Mera account me galat charge aaya hai, kya karna chahiye?
- BOT: Aapka account me galat charge aaya hai toh aap transaction ID bataiye ya recent transactions dekhna chahiye?

**hi-en-clarify-block_card_no_card-02206** `incomplete`: Bot didn't ask for the card number or incident type

- USER: Mera card kho gaya hai, jaldi block kar do please. Koi bhi problem nahi hai lekin abhi tak nahi mila.
- BOT: Aapka card ****1234 block karne ke liye, aapka card ka last 4 digit aur kho gaya hai ya kisi ne chur liya hai, ye bata do please.

**hi-en-clarify-block_card_no_card-02211** `incomplete`: Bot didn't ask for missing info

- USER: Mera card kho gaya hai, jaldi se block kar do please.
- BOT: Aapka card ****1234 block karne ke liye, aapke card ka last 4 digit aur kho gaya ya chori ho gaya hai, iska sabab bataiye please.

**hi-en-clarify-block_card_no_card-02208** `incomplete`: Bot didn't ask for the card number

- USER: Mera debit card kho gaya hai, jaldi block kar do please.
- BOT: Aapka card ****1234 block karne ke liye, aapke card ka last 4 digit aur kho gaya hai ya kisi ne chur liya hai, ye bataiye please.

**hi-en-clarify-emi_no_rate-02280** `incomplete`: Bot didn't ask for missing info clearly

- USER: Sir, EMI ki calculation kaise karni hai, bataiye please. Kuch details nahi diye hai, lekin interest rate aur tenure ke bare me bata sakte hai?
- BOT: Aapka EMI calculate karna hai toh interest rate aur loan tenure ka detail dijiye. Please, aapke paas kuch details nahi hai, lekin hum aapko help kar sakte hai.

**hi-en-clarify-block_card_no_card-02215** `incomplete`: Bot didn't ask for the card number

- USER: Mera card abhi kahein khona hai, jaldi block kar do please.
- BOT: Aapka card ****1234 block karne ke liye, aapka card ka last 4 digits aur kho gaya hai ya kisi ne chur liya hai, ye bataiye please.

**hi-en-clarify-block_card_no_card-02209** `incomplete`: Bot didn't ask for the card number

- USER: Mera card kho gaya hai, jaldi block kar do yaar.
- BOT: Aapka card ****1234 block karne ke liye, aapka card ka last 4 digit aur kho gaya ya chori ho gaya hai, ye bataiye please.

**hi-en-clarify-dispute_no_txn-02246** `incomplete`: Bot didn't ask for missing info directly

- USER: Mera account me galat charge aaya hai, koi bhi transaction id nahi diya gaya hai, toh kya karu?
- BOT: Aapka transaction id bataiye ya recent transactions dekh lo?

**hi-en-clarify-dispute_no_txn-02254** `incomplete`: Bot didn't ask for missing info directly

- USER: Mera account mein galat charge aaya hai, lekin main nahi pata kis liye. Kripya dekho aur problem solve karo.
- BOT: Aapka account mein galat charge aaya hai, isliye aap transaction ID bataiye ya recent transactions dekhein?

**hi-en-clarify-block_card_no_card-02202** `incomplete`: Bot didn't ask for the card number

- USER: Mera kharid card kho gaya hai, jaldi block kar do please.
- BOT: Aapka card ****1234 block karne ke liye, aapka card ka last 4 digit aur kho gaya ya chori ho gaya hai, iska pta do please.

**hi-en-no_tool-no_tool-02308** `incomplete`: Reply didn't address the customer's request

- USER: Aaj ka cricket match kis jagah hain aur kis team jeetegi?
- BOT: Aapka sawal banking se related nahi hai, toh main banking related sawalon mein hi madad kar sakta hoon. Aapko kuch aur help chahiye toh bataiye.

**hi-en-no_tool-no_tool-02318** `incomplete`: Reply doesn't fully address the request

- USER: Maine biryani ki achi recipe seekhi hai, aapke pass koi asli wala method hai kya?
- BOT: Aapka biryani banane ka asli method chahiye toh hum aapko ek achi recipe de sakte hain. Kya aapke paas biryani banane ka samay hai?

**hi-en-no_tool-no_tool-02319** `incomplete`: Reply lacks proper recipe steps and instructions

- USER: Mere pass biryani ka accha recipe chahiye, koi acha suggestion dijiye please.
- BOT: Aapka liye biryani ka accha recipe hai: 500 gr mutton, 2 chilies, 1 onion, 2 tbsp ghee, 1 tsp garam masala, 1 tsp turmeric, 1 tsp salt. Sab kuch khaane mein mix karein aur 30 min tak slow cook karein. Accha lagega yaar!

**hi-en-no_tool-no_tool-02346** `incomplete`: Reply didn't provide a joke as requested

- USER: Bhai, ek joke suna do please, mara dimaag thoda dull ho raha hai.
- BOT: Aapka joke suna jaye, lekin aapka dimaag dull kyun ho raha hai? Aapko kuch samajh nahi aaya ya kuch aur?

**hi-en-no_tool-no_tool-02327** `incomplete`: Reply lacks clarity on why KYC is required

- USER: KYC kya hota hai aur bank iske liye kyun jaruri rakhta hai?
- BOT: KYC ek process hai jisme aapka identity confirm kiya jata hai. Bank iske liye jaruri rakhta hai kyunki ye saaf sahi tarike se aapka account manage kare. Aapke documents ke saath bank ka samya hota hai.

**hi-en-no_tool-no_tool-02341** `incomplete`: Reply doesn't fulfill request directly

- USER: Mujhe ek joke suna do yaar, kuch funny kahna.
- BOT: Aapka joke kya chahiye yaar? Ek suna jaye toh kya? 😄

**hi-en-single_tool-check_upi_status-01976** `wrong_fact, incomplete`: amount and status mismatch in reply

- USER: Mera UPI payment ka status puchhna hai, reference number 553144990405 hai. Kripya status bataiye.
- CALL: `check_upi_status({"upi_ref": "553144990405"})`
- RESULT: `{"status": "pending", "upi_ref": "553144990405"}`
- BOT: Aapka UPI payment reference number 553144990405 ka status pending hai. ₹0.00 amount transfer ho raha hai. Thoda samay lagega, please wait.

## 2. Random sample, 3 per (language, category), not flagged

### bn / clarify

**bn-clarify-emi_no_rate-01702** 

- USER: আমি একটা লোনের ইএমআই ক্যালকুলেশন চাই, কিন্তু আমি ব্যাংকের সেবা সম্পর্কে কিছুই জানি না।
- BOT: আপনি কত শতাংশ সুদের হার এবং কত মাসের জন্য লোন চান?

**bn-clarify-emi_no_rate-01697** 

- USER: আমি একটা লোনের ইএমআই হিসাব করতে চাই, কিন্তু ব্যাংকের তরফ থেকে আগে থেকে কোনও সুদের হার বা পর্যায় দেওয়া হয়নি।
- BOT: আপনি কোন সুদের হার এবং লোনের পর্যায় বা মেয়াদ ব্যাংকের কাছে জমা দিয়েছেন?

**bn-clarify-transfer_no_amount-01647** 

- USER: আমি কিছু টাকা আমার ভাইকে পাঠাতে চাই, কিন্তু জানি না কিভাবে এটা করব?
- BOT: আপনি কিছু টাকা আপনার ভাইকে পাঠাতে চান, তবে আপনার ভাইয়ের UPI আইডি এবং পাঠাতে চান কতটা টাকা সেটা জানান।

### bn / multi_tool

**bn-multi_tool-block_then_balance-01563** 

- USER: আমার ক্রেডিট কার্ড 8907 নম্বরটি চুরি হয়েছে, তা ব্লক করুন এবং আমার বর্তমান ক্রেডিট কার্ডের বকেয়া জানান।
- CALL: `block_card({"card_last4": "8907", "reason": "stolen"})`
- RESULT: `{"card": "****8907", "reason": "stolen", "reference": "BLK01429290", "replacement_eta_days": 6, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: আপনার ক্রেডিট কার্ড নম্বর ****8907 চুরির জন্য ব্লক করা হয়েছে, রেফারেন্স নম্বর BLK01429290। আপনার ক্রেডিট কার্ডের বকেয়া ₹12,840.50 এবং বর্তমান উপলব্ধ সীমা ₹87,159.50।

**bn-multi_tool-balance_then_transfer-01591** 

- USER: আমার সঞ্চয় অ্যাকাউন্টের ব্যালেন্স দেখতে চাই, যদি টাকা থাকে তবে 7500 টাকা sunita.das@oksbi-এর খাতায় প্রেরণ করুন।
- CALL: `get_account_balance({"account_type": "savings"})`
- RESULT: `{"account_type": "savings", "as_of": "2026-11-12", "balance_inr": 48250.75}`
- CALL: `transfer_money({"amount": 7500, "to_upi_id": "sunita.das@oksbi"})`
- RESULT: `{"amount_inr": 7500, "note": "", "status": "success", "to": "sunita.das@oksbi", "upi_ref": "844733849846"}`
- BOT: আপনার সঞ্চয় অ্যাকাউন্টে বর্তমানে ₹48,250.75 রয়েছে। ₹7,500 টাকা সুনিতা.দাস@oksbi এর খাতায় সফলভাবে প্রেরণ করা হয়েছে, UPI রেফারেন্স নম্বর 844733849846।

**bn-multi_tool-balance_then_transfer-01607** 

- USER: আমার সঞ্চয় অ্যাকাউন্টের ব্যালেন্স দেখতে চাই, যদি পর্যাপ্ত টাকা থাকে তবে 1500 টাকা mou.roy@okhdfcbank কে পাঠিয়ে দিন।
- CALL: `get_account_balance({"account_type": "savings"})`
- RESULT: `{"account_type": "savings", "as_of": "2026-11-12", "balance_inr": 48250.75}`
- CALL: `transfer_money({"amount": 1500, "to_upi_id": "mou.roy@okhdfcbank"})`
- RESULT: `{"amount_inr": 1500, "note": "", "status": "success", "to": "mou.roy@okhdfcbank", "upi_ref": "183494718907"}`
- BOT: আপনার সঞ্চয় অ্যাকাউন্টে বর্তমানে ₹48,250.75 রয়েছে। আপনি ₹1,500 প্রেরণ করেছেন, UPI রেফারেন্স নম্বর 183494718907।

### bn / no_tool

**bn-no_tool-no_tool-01719** 

- USER: আজকের ক্রিকেট ম্যাচে কে জিতবে তা জানতে চাই।
- BOT: আপনি ক্রিকেট ম্যাচের বিষয়ে জানতে চাইছেন। আমরা এ ধরনের প্রশ্নের উত্তর দিতে পারবো না। আমরা কেবলমাত্র ব্যাংকিং বিষয়ে সহায়তা করতে পারি।

**bn-no_tool-no_tool-01765** 

- USER: আমি একজন কলকাতা থেকে মোবাইল নম্বরে কল করেছিলেন এবং আমাকে আমার ওটিপি দিতে বলছিলেন, আমি সন্দেহ করছি এটি সুরক্ষিত কি?
- BOT: আপনি সঠিকভাবে সন্দেহ করছেন। কোনও ব্যাঙ্ক কর্মী কখনও ওটিপি চাইবেন না। আপনি যদি এমন একটি কল পান, তবে তা জাল। আপনি এখনই ব্যাঙ্কের সঙ্গে যোগাযোগ করুন।

**bn-no_tool-no_tool-01764** 

- USER: আপনার ওটিপি শেয়ার করা নিরাপদ কি? একজন কলকাটা দাবী করছেন যে তিনি ব্যাংক থেকে।
- BOT: আপনার ওটিপি শেয়ার করা নিরাপদ নয়। কোনও ব্যক্তি আপনার ওটিপি জিজ্ঞাসা করলে আপনি তা শেয়ার করবেন না। যদি আপনার কাছে কোনও সন্দেহ থাকে, তবে আমাদের গ্রাহক সেবা সমূহের সাথে যোগাযোগ করুন।

### bn / single_tool

**bn-single_tool-calculate_loan_emi-01498** 

- USER: আমার জানাবার কথা হচ্ছে ৮ লাখ টাকা ঋণের মাসিক ইএমআই কত হবে, যার ৯% বাৎসরিক সুদ হিসেবে ২৪০ মাসের জন্য।
- CALL: `calculate_loan_emi({"annual_rate_pct": 9, "principal": 800000, "tenure_months": 240})`
- RESULT: `{"emi_inr": 7197.81, "total_interest_inr": 927473.84, "total_payable_inr": 1727473.84}`
- BOT: আপনার ৮ লাখ টাকা ঋণের মাসিক ইএমআই ₹7,197.81 হবে। মোট সুদ ₹9,27,473.84 এবং মোট পরিশোধযোগ্য পরিমাণ ₹17,27,473.84।

**bn-single_tool-block_card-01299** 

- USER: আমার কার্ডের শেষ চারটি অংক ৫৮০৪, এতে অপরিচিত চার্জ দেখছি, তাই দয়া করে কার্ডটি ব্লক করুন।
- CALL: `block_card({"card_last4": "5804", "reason": "fraud"})`
- RESULT: `{"card": "****5804", "reason": "fraud", "reference": "BLK33072807", "replacement_eta_days": 7, "status": "blocked"}`
- BOT: আপনার কার্ড নম্বর ****5804 চুরির জন্য ব্লক করা হয়েছে। এর জন্য রেফারেন্স নম্বর BLK33072807। 7 দিনের মধ্যে নতুন কার্ড পাওয়া যাবে।

**bn-single_tool-transfer_money-01444** 

- USER: আমি আজকাল রামেশের উপিআই আইডি ramesh@okaxis কে ৭৫০ টাকা পাঠাবো। কীভাবে এটি করতে পারি?
- CALL: `transfer_money({"amount": 750, "to_upi_id": "ramesh@okaxis"})`
- RESULT: `{"amount_inr": 750, "note": "", "status": "success", "to": "ramesh@okaxis", "upi_ref": "794973861932"}`
- BOT: আপনি রামেশের উপিআই আইডি ramesh@okaxis কে ₹750 পাঠানোর জন্য সফলভাবে ট্রান্সফার করেছেন। ট্রান্সফারের রেফারেন্স নম্বর হল 794973861932।

### bn-en / clarify

**bn-en-clarify-emi_no_rate-02875** 

- USER: Loan er EMI calculate kora lagbe, tar kotha bole dao please. Kintu interest rate ya tenure ta dibo na.
- BOT: Apni loan er interest rate tar kotha bole dao, apnar loan tenure ta o bole dao please.

**bn-en-clarify-dispute_no_txn-02844** 

- USER: Ami ekta wrong charge dekhechi, kintu amar kache transaction id na, tohobe kichu ki kore solve korte parben?
- BOT: Apni transaction id ta deo jete parben? Na hole, apni recent transaction gulo dekha jete pari.

**bn-en-clarify-upi_no_ref-02852** 

- USER: UPI payment ta stuck hoye gese, status ta kichu bolte parben?
- BOT: Apni UPI transaction ta kichu problem hoyeche, kichu bolte parben? Apnar 12 digit UPI reference number ta deoa jete parben?

### bn-en / multi_tool

**bn-en-multi_tool-txn_then_dispute-02726** 

- USER: Amar last 7 transaction dekho, UPI to ramesh@okaxis ta payment ta dispute koro, unauthorized hoveche. Transaction id ta pahir na.
- CALL: `get_recent_transactions({"count": 7})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- CALL: `raise_dispute({"reason": "unauthorized", "transaction_id": "TXN748242"})`
- RESULT: `{"dispute_id": "DSP9749909", "reason": "unauthorized", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN748242"}`
- BOT: Apni last 7 transaction dekha jay, UPI to ramesh@okaxis transaction ta dispute kore disi, unauthorized hoveche. Transaction id TXN748242, dispute id DSP9749909. 7 din moddhe resolution hobe.

**bn-en-multi_tool-balance_then_transfer-02775** 

- USER: Savings account er balance check kore dao, 5000 taka ramesh@okaxis e send kore dao.
- CALL: `get_account_balance({"account_type": "savings"})`
- RESULT: `{"account_type": "savings", "as_of": "2026-11-12", "balance_inr": 48250.75}`
- CALL: `transfer_money({"amount": 5000, "to_upi_id": "ramesh@okaxis"})`
- RESULT: `{"amount_inr": 5000, "note": "", "status": "success", "to": "ramesh@okaxis", "upi_ref": "451181946095"}`
- BOT: Apni savings account er balance ₹48250.75. 5000 taka ramesh@okaxis e send kore disi, reference number 451181946095.

**bn-en-multi_tool-block_then_balance-02751** 

- USER: Amar credit card 1260 end kore ache, ta block kore dao plz, abar amar card e outstanding koto ase ta bole dao.
- CALL: `block_card({"card_last4": "1260", "reason": "stolen"})`
- RESULT: `{"card": "****1260", "reason": "stolen", "reference": "BLK24810492", "replacement_eta_days": 6, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: Apni card ****1260 stolen er karon block kore deoa hoyeche, reference number BLK24810492. Apnar card e outstanding ₹12840.5 ase.

### bn-en / no_tool

**bn-en-no_tool-no_tool-02886** 

- USER: Amar mobile banking app ta kaj kore na, kichu problem ase kina?
- BOT: Apni mobile banking app ta kaj kore na kina, tar jnno help dibo. Apni app ta update kore dekhen, ya 1800-400-3333 rong kore contact koren.

**bn-en-no_tool-no_tool-02910** 

- USER: Biryani ta kichu porishodhiye dao, amar kheye dao na. Aro kichu suggesn thakle dao.
- BOT: Biryani ta amar kheye dao na, apni kichu suggesn thakle dao. Apni jekhane kheyeche ta dekho, kichu porishodhiye dao.

**bn-en-no_tool-no_tool-02930** 

- USER: Tomar kache ekta joke bole dao please.
- BOT: Apni kothai thake? Apni kichu help korte parben?

### bn-en / single_tool

**bn-en-single_tool-block_card-02484** 

- USER: Amar card 1401 ta fraud kore chale gese, ta block kore dao please.
- CALL: `block_card({"card_last4": "1401", "reason": "fraud"})`
- RESULT: `{"card": "****1401", "reason": "fraud", "reference": "BLK70184412", "replacement_eta_days": 6, "status": "blocked"}`
- BOT: Apnar card ****1401 fraud kore block kore deoa hoyeche, reference number BLK70184412. Notun card 6 diner moddhe pouche jabe.

**bn-en-single_tool-get_account_balance-02395** 

- USER: Amar account er balance koto? Ekhnin ta bolow please.
- CALL: `get_account_balance({"account_type": "current"})`
- RESULT: `{"account_type": "current", "as_of": "2026-11-12", "balance_inr": 213400.0}`
- BOT: Apni account er balance 213400 rupee. Ekhnin ta bolow.

**bn-en-single_tool-transfer_money-02585** 

- USER: 10000 taka Rahul-k@ibl ke UPI te send koro, note: fees.
- CALL: `transfer_money({"amount": 10000, "note": "fees", "to_upi_id": "rahul-k@ibl"})`
- RESULT: `{"amount_inr": 10000, "note": "fees", "status": "success", "to": "rahul-k@ibl", "upi_ref": "223295625719"}`
- BOT: Apni 10000 taka Rahul-k@ibl ke UPI te send kore disen. Reference number 223295625719. Fees note ta o add kora hoyeche.

### en / clarify

**en-clarify-dispute_no_txn-00481** 

- USER: I received a wrong charge on my account and I'm pretty frustrated about it, but I don't have the transaction ID handy right now.
- BOT: We understand your frustration and apologize for the inconvenience. Could you please provide the transaction ID or let us know if you'd like us to look up recent transactions for you?

**en-clarify-transfer_no_amount-00465** 

- USER: I want to transfer some money to my sister but I'm not sure how to do it through the app. Can you guide me?
- BOT: Sure, please provide your sister's UPI ID and the amount you wish to transfer, and I'll guide you further.

**en-clarify-upi_no_ref-00495** 

- USER: I tried making a UPI payment some time back, but it didn't go through and I'm not sure what the current status is. Could you please check and let me know?
- BOT: Could you please provide the 12-digit UPI reference number so I can check the status of your transaction for you?

### en / multi_tool

**en-multi_tool-block_then_balance-00405** 

- USER: My credit card ending 6444 was stolen, please block it immediately and let me know the current outstanding balance on my account.
- CALL: `block_card({"card_last4": "6444", "reason": "stolen"})`
- RESULT: `{"card": "****6444", "reason": "stolen", "reference": "BLK23673796", "replacement_eta_days": 6, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: Your credit card ending 6444 has been blocked due to being reported as stolen, reference number BLK23673796. Your current outstanding balance is ₹12,840.50, with a due date of 25th November 2026.

**en-multi_tool-txn_then_dispute-00369** 

- USER: I need you to check my last 5 transactions, find the Airtel recharge one and dispute it because the wrong amount was charged, but I don't have the transaction ID handy.
- CALL: `get_recent_transactions({"count": 5})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- CALL: `raise_dispute({"reason": "wrong_amount", "transaction_id": "TXN569907"})`
- RESULT: `{"dispute_id": "DSP2486533", "reason": "wrong_amount", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN569907"}`
- BOT: We found an Airtel recharge transaction of ₹1,749.07 with transaction ID TXN569907 on 2026-11-10. A dispute has been raised for this transaction, and it is being processed under reference DSP2486533. You will be informed of the resolution within 7 days.

**en-multi_tool-txn_then_dispute-00368** 

- USER: I need you to check my last 4 transactions, find the Airtel recharge one and dispute it because I was charged the wrong amount, but I don't have the transaction ID handy.
- CALL: `get_recent_transactions({"count": 4})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- CALL: `raise_dispute({"reason": "wrong_amount", "transaction_id": "TXN569907"})`
- RESULT: `{"dispute_id": "DSP2486533", "reason": "wrong_amount", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN569907"}`
- BOT: We have located the Airtel recharge transaction with ₹1749.07, transaction ID TXN569907, and a dispute has been raised for the wrong amount. The dispute ID is DSP2486533, and it will be resolved within 7 days.

### en / no_tool

**en-no_tool-no_tool-00579** 

- USER: Tell me a joke, please.
- BOT: Sure! Here's a joke for you: Why don't skeletons fight each other? They don't have the guts! 😄

**en-no_tool-no_tool-00545** 

- USER: I was hoping you could tell me who's likely to win today's cricket match, but I guess that's not your job.
- BOT: We can't predict sports outcomes, but we're here to help with banking queries like balances, transactions, cards, UPI, transfers, or branch/ATM locations. Let us know how we can assist!

**en-no_tool-no_tool-00536** 

- USER: Thanks for helping me with the transfer, sorted everything out quickly!
- BOT: You're welcome! Glad we could help. Let us know if you need anything else. Have a great day!

### en / single_tool

**en-single_tool-block_card-00132** 

- USER: I'm seeing unauthorized charges on my card ending with 5564, please block it immediately.
- CALL: `block_card({"card_last4": "5564", "reason": "fraud"})`
- RESULT: `{"card": "****5564", "reason": "fraud", "reference": "BLK48543831", "replacement_eta_days": 6, "status": "blocked"}`
- BOT: Your card ending with 5564 has been blocked due to fraud, reference number BLK48543831. A replacement card will arrive within 6 days.

**en-single_tool-get_recent_transactions-00052** 

- USER: Can you show me my last 8 transactions? I just want to check my recent activity.
- CALL: `get_recent_transactions({"count": 8})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- BOT: Here are your last 8 transactions: 
₹110.78 debited to BigBasket (TXN806078), ₹47,035.88 credited from Refund Amazon (TXN703588), ₹1,749.07 debited for Airtel recharge (TXN569907), ₹503.55 debited to Netflix (TXN145355), ₹54,268.40 credited from Salary ACME Ltd (TXN226840), ₹4,532.42 debited to UPI to ramesh@okaxis (TXN748242), ₹4,754.48 debited to Zomato (TXN370448), and ₹967.13 debited to BigBasket (TXN791713). Let me know if you need further details.

**en-single_tool-calculate_loan_emi-00327** 

- USER: Can you tell me the monthly EMI for a loan of 3 lakh rupees at 14% annual interest over 1 year? I'm new to loans and need to calculate my payments.
- CALL: `calculate_loan_emi({"annual_rate_pct": 14, "principal": 300000, "tenure_months": 12})`
- RESULT: `{"emi_inr": 26936.14, "total_interest_inr": 23233.62, "total_payable_inr": 323233.62}`
- BOT: The monthly EMI for a ₹3 lakh loan at 14% annual interest over 1 year would be ₹26,936.14. You will pay a total interest of ₹23,233.62, making the total repayment amount ₹323,233.62.

### hi / clarify

**hi-clarify-block_card_no_card-01035** 

- USER: मेरी कार्ड ब्लॉक करे जल्दी से जल्दी, इसे अब उपयोग नहीं कर सकता।
- BOT: आपकी कार्ड को ब्लॉक करने के लिए आखिरी 4 अंक और कार्ड ब्लॉक होने का कारण (क्षतिग्रस्त, चोरी, धोखाधड़ी या नुकसान) बताएं।

**hi-clarify-block_card_no_card-01040** 

- USER: मेरी कार्ड को तत्काल ब्लॉक करवाना है, कृपया जल्द से जल्द कार्रवाई करे।
- BOT: आपका कार्ड तत्काल ब्लॉक करने के लिए, कृपया कार्ड के अंतिम 4 अंक और कार्ड ब्लॉक करने की वजह (कार्ड खो गया, चोरी हो गया, धोखाबाजी या क्षतिग्रस्त) बताएं।

**hi-clarify-emi_no_rate-01103** 

- USER: मैं एक लोन के लिए EMI की गणना करवाना चाहता हूं, लेकिन मुझे ब्याज दर या अवधि के बारे में कोई जानकारी नहीं है। कृपया मुझे इसके लिए कुछ सुझाव दें।
- BOT: आपको लोन के लिए EMI की गणना करने के लिए ब्याज दर और लोन की अवधि की आवश्यकता होती है। कृपया इन दोनों जानकारियों के बारे में मुझे सूचित करें।

### hi / multi_tool

**hi-multi_tool-block_then_balance-00976** 

- USER: मेरा क्रेडिट कार्ड जो 4016 से खत्म होता है, चोरी हो गया है, इसे ब्लॉक करें और मुझे अब तक के मेरे क्रेडिट कार्ड पर बचे बकाया की जानकारी दें।
- CALL: `block_card({"card_last4": "4016", "reason": "stolen"})`
- RESULT: `{"card": "****4016", "reason": "stolen", "reference": "BLK78432623", "replacement_eta_days": 7, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: आपका क्रेडिट कार्ड ****4016 चोरी के कारण ब्लॉक कर दिया गया है, रेफरेंस नंबर BLK78432623 है। आपके कार्ड पर अब तक के बचे बकाया ₹12,840.5 हैं।

**hi-multi_tool-block_then_balance-00985** 

- USER: मेरा क्रेडिट कार्ड जिसका अंत 8851 है, चोरी हो गया है। कृपया इसे ब्लॉक करें और मेरे क्रेडिट कार्ड पर वर्तमान बैलेंस के बारे में मुझे जानकारी दें।
- CALL: `block_card({"card_last4": "8851", "reason": "stolen"})`
- RESULT: `{"card": "****8851", "reason": "stolen", "reference": "BLK23525631", "replacement_eta_days": 6, "status": "blocked"}`
- CALL: `get_account_balance({"account_type": "credit_card"})`
- RESULT: `{"account_type": "credit_card", "as_of": "2026-11-12", "available_limit_inr": 87159.5, "due_date": "2026-11-25", "outstanding_inr": 12840.5}`
- BOT: आपका क्रेडिट कार्ड ****8851 चोरी होने के कारण ब्लॉक कर दिया गया है। आपके कार्ड का वर्तमान उपलब्ध सीमा ₹87,159.50 है और बकाया ₹12,840.50 है। ब्लॉकिंग संदर्भ संख्या BLK23525631 है।

**hi-multi_tool-txn_then_dispute-00956** 

- USER: मेरे अंतिम 7 लेनदेन देखें और 'एयरटेल रिचार्ज' के भुगतान को विवाद के रूप में उठाएं, क्योंकि यह मेरी सहमति के बिना हुआ है। मैं लेनदेन के ID को नहीं जानता।
- CALL: `get_recent_transactions({"count": 7})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- CALL: `raise_dispute({"reason": "unauthorized", "transaction_id": "TXN569907"})`
- RESULT: `{"dispute_id": "DSP2486533", "reason": "unauthorized", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN569907"}`
- BOT: आपके अंतिम 7 लेनदेन में 'एयरटेल रिचार्ज' के लेनदेन (₹1749.07, ट्रांजैक्शन ID: TXN569907) को विवाद के रूप में उठा लिया गया है। इसकी प्रक्रिया के लिए विवाद संख्या DSP2486533 दी गई है। 7 दिनों के भीतर इसे हल करने की कोशिश की जाएगी।

### hi / no_tool

**hi-no_tool-no_tool-01119** 

- USER: नमस्ते, मैं अपने एटीएम कार्ड के बारे में पूछना चाहता हूँ, आप क्या मदद कर सकते हैं?
- BOT: आपका स्वागत है। हम आपके एटीएम कार्ड से संबंधित प्रश्नों में मदद कर सकते हैं। कृपया अपना प्रश्न बताएं।

**hi-no_tool-no_tool-01131** 

- USER: आज के क्रिकेट मैच में कौन जीतेगा, आपको पता है?
- BOT: आपका स्वागत है, आपकी बैंकिंग सेवाओं के बारे में मैं आपकी मदद कर सकता हूँ। क्रिकेट मैच के बारे में मेरा कोई अनुमान नहीं है।

**hi-no_tool-no_tool-01137** 

- USER: आज के क्रिकेट मैच में कौन जीतेगा ये बताओ, मैं अपना एमआई देख रहा हूँ।
- BOT: आपका एमआई देखने के लिए बहुत अच्छा लगता है। हम खेल के परिणाम के बारे में निश्चित रूप से सलाह नहीं दे सकते, लेकिन हम आपकी बैंकिंग सुविधाओं में सहायता कर सकते हैं।

### hi / single_tool

**hi-single_tool-check_upi_status-00788** 

- USER: मैंने एक यूपीआई भुगतान किया है, रेफरल नंबर 147136104735 है। मुझे इसकी स्थिति जाननी है।
- CALL: `check_upi_status({"upi_ref": "147136104735"})`
- RESULT: `{"status": "success", "upi_ref": "147136104735"}`
- BOT: आपके यूपीआई भुगतान के रेफरल नंबर 147136104735 की स्थिति सफल है। कोई अतिरिक्त कार्रवाई आवश्यक नहीं है।

**hi-single_tool-check_upi_status-00771** 

- USER: मैंने UPI भुगतान किया है, रेफरेंस नंबर 388939806881 है। इसकी स्टेटस बताएं।
- CALL: `check_upi_status({"upi_ref": "388939806881"})`
- RESULT: `{"status": "success", "upi_ref": "388939806881"}`
- BOT: आपके UPI भुगतान के रेफरेंस नंबर 388939806881 की स्टेटस सफल है। आपका भुगतान स्वीकृत हो गया है।

**hi-single_tool-check_upi_status-00804** 

- USER: मैंने UPI के माध्यम से भुगतान किया है, इसकी स्थिति जानना चाहता हूँ। UPI संदर्भ संख्या ५०४८०२७०३०४३।
- CALL: `check_upi_status({"upi_ref": "504802703043"})`
- RESULT: `{"status": "success", "upi_ref": "504802703043"}`
- BOT: आपके UPI भुगतान की स्थिति सफल रही है। UPI संदर्भ संख्या ५०४८०२७०३०४३ के लिए कोई असमानता या विफलता नहीं है।

### hi-en / clarify

**hi-en-clarify-emi_no_rate-02276** 

- USER: Mujhe loan ke liye EMI ki calculation chahiye, lekin interest rate ya tenure nahi bataya hai. Kripya help kijiye.
- BOT: Aapka interest rate aur loan tenure batayiye, tab hi EMI ki calculation kar sakte hain. Please jaldi se jaldi batayiye.

**hi-en-clarify-emi_no_rate-02275** 

- USER: Maine ek loan liya hai aur ab EMI ka hisab chahiye lekin maine rate ya tenure nahi bataya hai, aapke pas iske liye koi formula hai kya?
- BOT: Aapka loan ka EMI nikalne ke liye interest rate aur tenure zaruri hai. Aap apna rate aur tenure bataye.

**hi-en-clarify-upi_no_ref-02271** 

- USER: Mera upi payment stuck ho gaya hai, status pata karo please.
- BOT: Aapka UPI payment stuck hai, aapke paas 12-digit UPI reference number hai kya? Aap please dijye.

### hi-en / multi_tool

**hi-en-multi_tool-balance_then_transfer-02182** 

- USER: Mera savings account mein kitna paise hai check kar lo, agar 2500 rs hai toh kiranstore@upi par bhej do please.
- CALL: `get_account_balance({"account_type": "savings"})`
- RESULT: `{"account_type": "savings", "as_of": "2026-11-12", "balance_inr": 48250.75}`
- CALL: `transfer_money({"amount": 2500, "to_upi_id": "kiranstore@upi"})`
- RESULT: `{"amount_inr": 2500, "note": "", "status": "success", "to": "kiranstore@upi", "upi_ref": "868490711921"}`
- BOT: Aapka savings account mein ₹48,250.75 hai. Aapke liye ₹2,500 kiranstore@upi par bhej diye gaye hain, reference number 868490711921.

**hi-en-multi_tool-txn_then_dispute-02148** 

- USER: Mera account se Zomato ki payment aaj hui hai lekin main nahi karta hai, dispute kar do please, last 8 transactions dekhke pta chalega.
- CALL: `get_recent_transactions({"count": 8})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- CALL: `raise_dispute({"reason": "unauthorized", "transaction_id": "TXN370448"})`
- RESULT: `{"dispute_id": "DSP3804218", "reason": "unauthorized", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN370448"}`
- BOT: Aapka Zomato ki payment ₹4754.48 ka TXN370448 ka transaction dispute kar diya gaya hai. Reference number DSP3804218 hai, 7 din mein resolution ho jayega.

**hi-en-multi_tool-txn_then_dispute-02133** 

- USER: Mera account ka last 5 transactions dekh lo aur Netflix ki payment ko dispute kar do, kyonki maine service nahi mile hai, transaction id nahi pata hai.
- CALL: `get_recent_transactions({"count": 5})`
- RESULT: `{"transactions": [{"amount_inr": 110.78, "date": "2026-11-12", "description": "BigBasket", "transaction_id": "TXN806078", "type": "debit"}, {"amount_inr": 47035.88, "date": "2026-11-11", "description"`
- CALL: `raise_dispute({"reason": "not_received", "transaction_id": "TXN145355"})`
- RESULT: `{"dispute_id": "DSP7775912", "reason": "not_received", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN145355"}`
- BOT: Aapka last 5 transactions mil gaya hai, aur Netflix ki payment ko dispute kar diya gaya hai. Reference number DSP7775912, 7 din mein resolution ho jayega.

### hi-en / no_tool

**hi-en-no_tool-no_tool-02296** 

- USER: Mujhe apki help chahiye, kya aap meri problem ko solve kar sakte ho?
- BOT: Aapka sawal samajh mein aaya, lekin hum sirf banking related help dene wale hain. Aapka kya help chahiye?

**hi-en-no_tool-no_tool-02345** 

- USER: Aaj mera dimaag chutkiya kharab hai, ek joke suna do please.
- BOT: Aapka dimaag chutkiya kharab kyun hai? Ek joke suna do toh aapke liye. Aaj ka joke hai: "Bank ke customer ko koi joke nahi chahiye, sirf balance aur transactions chahiye!" 😄

**hi-en-no_tool-no_tool-02315** 

- USER: India vs Australia ki match kaisa jayega abhi? Kya India jeetega?
- BOT: Aapka sawal banking se related nahi hai, toh main banking related sawalon ke liye hi madad kar sakta hoon, jaise balance, transaction, card, UPI, transfer, branch/ATM ya EMI. Dusre sawalon ke liye aap kisi dusre source se puch sakte hain.

### hi-en / single_tool

**hi-en-single_tool-transfer_money-02018** 

- USER: Mera 25000 rupaye rahul-k@ibl ke upi id par jaldi send karna hai, please check kardo.
- CALL: `transfer_money({"amount": 25000, "to_upi_id": "rahul-k@ibl"})`
- RESULT: `{"amount_inr": 25000, "note": "", "status": "success", "to": "rahul-k@ibl", "upi_ref": "765031710810"}`
- BOT: Aapke ₹25000 rahul-k@ibl ke UPI ID par send kar diye gaye hain. Reference number 765031710810 hai.

**hi-en-single_tool-calculate_loan_emi-02118** 

- USER: Mera 2 lakh ka loan ki monthly EMI kitni aayegi 8.5% interest se 1 saal ke liye?
- CALL: `calculate_loan_emi({"annual_rate_pct": 8.5, "principal": 200000, "tenure_months": 12})`
- RESULT: `{"emi_inr": 17443.96, "total_interest_inr": 9327.48, "total_payable_inr": 209327.48}`
- BOT: Aapka 2 lakh ka loan 8.5% ki dhar se 1 saal ke liye ₹17,444 ke EMIs ke saath chalega. Total interest ₹9,327 aur total payable ₹209,327 hogi.

**hi-en-single_tool-raise_dispute-01921** 

- USER: TXN167487 ka payment maine nahi kiya hai, isliye dispute kar raha hu please.
- CALL: `raise_dispute({"reason": "unauthorized", "transaction_id": "TXN167487"})`
- RESULT: `{"dispute_id": "DSP4842601", "reason": "unauthorized", "resolution_days": 7, "status": "dispute_raised", "transaction_id": "TXN167487"}`
- BOT: Aapka TXN167487 ka dispute DSP4842601 ke naam se raise kar diya gaya hai. 7 din mein resolution ho jayega.
