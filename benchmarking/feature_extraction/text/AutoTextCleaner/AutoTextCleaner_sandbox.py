# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._AutoTextCleaner.AutoTextCleaner import \
    AutoTextCleaner

import re




text = [
    "Trump has 90 days to do 150 trade deals. No one is buying it",
    "President Donald Trump’s 90-day pause on his “reciprocal” tariffs gives his administration just three months to strike enormously complex trade deals with dozens of countries.",
    "Stocks have whipsawed as volatility has spiked. And other markets are sending a clear message of deep skepticism that Trump can pull it off.",
    "US consumer sentiment is now worse than during Great Recession",
    "Live Updates China raises tariffs on US goods to 125%",
    "Analysis Why a US-China trade war could be catastrophic",
    "Trump is waiting for Xi to call. The Chinese see it differently",
    "sciutto china tariff analysis.jpg",
    "What this tweet says about the US-China trade war 1:22",
    "farmer 1.jpg",
    "‘I never thought I was going to lose this much money’: Trump voter amid tariffs 3:24",
    "Kevin O'Leary.jpg",
    "Kevin O’Leary says US has to train China ‘like a puppy’ 1:57",
    "This undated photo provided by CASA, an immigrant advocacy organization, in April 2025, shows Kilmar Abrego Garcia. (CASA via AP)",
    "Breaking News",
    "Trump administration won’t say where Abrego Garcia is. He was mistakenly deported to El Salvador last month",
    "Judge to decide on release of Palestinian activist and student in hearing that his attorneys say will have ‘momentous implications’",
    "‘Stop trying to glamorize the mission’: Megyn Kelly slams Kristi Noem over ICE photo ops 1:31",
    "Satellite imagery captured Hurricane Helene a few hours before it made landfall in Florida on September 26, 2024. Clouds and moisture from the storm stretched from the Caribbean nearly to Canada.",
    "Trump’s budget plan eviscerates weather and climate research",
    "Judge will halt Trump administration from ending humanitarian parole for people from four countries",
    "LIVE",
    "See how the NYSE is reacting after China's latest tariffs move",
    "See how the NYSE is reacting after China ups retaliatory tariffs on US",
    "FRANKLIN, TENNESSEE - SEPTEMBER 25: Yasmin Williams performs onstage during day two of the 2022 Pilgrimage Music & Cultural Festival on September 25, 2022 in Franklin, Tennessee. (Photo by Erika Goldring/Getty Images for Pilgrimage Music & Cultural Festival)",
    "Musician says she was left shocked by ‘bizarre’ emails from acting Kennedy Center director Richard Grenell",
    "111756_BookSigningTikTok thumbnail.jpg",
    "An author was sitting alone at book signing event, then a viral TikTok made him a bestseller 1:10",
    "1Chance-Encounters-Jenny-and-Jason.jpg",
    "She was ‘on the top of a mountain in the middle of the Himalayas.’ Then a mysterious man walked through her door",
    "More top stories",
    "PITUFFIK, GREENLAND - MARCH 28: US Vice President JD Vance (2nd-R) and second lady Usha Vance (2nd-L) tour the US military's Pituffik Space Base on March 28, 2025 in Pituffik, Greenland. The visit is viewed by Copenhagen and Nuuk as a provocation amid President Donald Trump's bid to annex the strategically-placed, resource-rich Danish territory. (Photo by Jim Watson - Pool/Getty Images) Susannah Meyers on left.",
    "US removes commander in Greenland following Vance’s controversial visit",
    "This is one of the biggest challenges for people dating on ‘Love on the Spectrum’ 1:01",
    "RFK Jr. claims new research effort will find cause of ‘autism epidemic’ by September",
    "Video shows confrontation after Homeland Security agent arrives at 19-year-old green card applicant’s house 2:32",
    "Tempers flare over Musk and DOGE as Democrats tread lightly on tariffs during CNN town hall",
    "The global baby problem is keeping Elon Musk up at night. Meet the people trying to solve it. 4:58",
    "Ukraine’s European allies say Russian aggression is pushing peace out of reach, as US envoy meets Putin",
    "Aviation",
    "3 people killed after small plane crashes on busy Boca Raton street",
    "Plane’s double diversion has passengers spending almost a full day on board",
    "A helicopter crash left a Siemens executive, his family and their pilot dead. Here’s what we know",
    "Clean Energy",
    "The Ford Motor Co. and SK Innovation Co. electric vehicle and battery manufacturing complex under construction near Stanton, Tennessee, US, on Tuesday, Sept. 20, 2022. The Ford Motor Co. and SK Innovation Co. $5.6 billion manufacturing complex, known as BlueOval City, is due to begin building electric F-Series pickup trucks and the batteries that power them in three years. Photographer: Houston Cofield/Bloomberg via Getty Images",
    "For Subscribers",
    "Republicans poised to help Trump kill thousands of manufacturing jobs in places that voted for him",
    "CNN Podcasts",
    "Steaks are displayed at a grocery store in New York on May 12, 2022.",
    "The power (and pitfalls) of a high-protein diet",
    "Mustafa Suleyman, CEO of Inflection AI, speaks during The Wall Street Journal's WSJ Tech Live Conference in Laguna Beach, California on October 17, 2023. (Photo by Patrick T. Fallon / AFP) (Photo by PATRICK T. FALLON/AFP via Getty Images)",
    "Can we trust AI, or the people building it?",
    "Asian women with short and black hair in casual wear, sitting on a wooden floor side the window with some light in dark tone room and holding a mobile phone. Maybe she's cheating something to somebody",
    "Can grief bots help us heal?",
    "What's Buzzing",
    "Watch the latest CNN Headlines",
    "A model walks the runway during the Armani Prive Haute Couture Fall/Winter 2024-2025 fashion show as part of Paris Fashion Week on June 25, 2024 in Paris, France. (Photo by Victor VIRGILE/Gamma-Rapho via Getty Images)",
    "Australian model dies at age 27",
    "Scott Shriner and Jillian Lauren attend PEN America 2018 LitFest Gala at the Beverly Wilshire Four Seasons Hotel on November 02, 2018 in Beverly Hills, California.",
    "Weezer bassist’s wife Jillian Shriner was recovering from cancer-related surgery before arrest",
    "Check these out",
    "STAFFORDSHIRE, UNITED KINGDOM - APRIL 05: A kingfisher sits atop a 'No Fishing' sign with a freshly caught fish in its beak at Teddesley Park in Staffordshire, United Kingdom on April 05, 2025. (Photo by Stuart Brock/Anadolu via Getty Images)",
    "Gallery The week in 40 photos",
    "manosphere dating thumb 2.jpg",
    "This woman dated only far-right men for a year: ‘They were so insecure’ 4:38",
    "Cropped shot of an unrecognizable woman using ingredients to make a healthy salad",
    "Mediterranean diet and exercise improve bone density in older women, study finds",
    "New York, NY - Selena Gomez and Benny Blanco wear matching coats as they leave after the Knicks game at Madison Square Garden in New York City Pictured: Selena Gomez, Benny Blanco BACKGRID USA 8 APRIL 2025 BYLINE MUST READ: Santi Ramales / BACKGRID USA: +1 310 798 9111 / usasales@backgrid.com UK: +44 208 344 2007 / uksales@backgrid.com *UK Clients - Pictures Containing Children Please Pixelate Face Prior To Publication*",
    "Look of the Week: Selena Gomez and Benny Blanco’s love language is matching looks",
    "For Subscribers",
    "Hope Pann's home in Mandalay, now demolished, after it was damaged from the earthquake.",
    "Some students from Myanmar are too scared to leave the US, even after their families were ravaged by disaster",
    "An attendee wears a shirt promoting higher birth rates is seen at the Natal Conference.",
    "Tech bros and tradwives are unlikely allies in a little-known movement that’s gaining momentum",
    "20250305-firing-quad-top-image.jpg",
    "Why the firing squad may be making a comeback",
    "CNN Underscored",
    "Best-in-class",
    "practical-mothers-day-cnnu.jpg",
    "She said no gifts, but these 54 practical Mother’s Day gifts will win her over",
    "Nap dresses really are that comfy. Here are 15 our editors and stylists love",
    "20 ingenious Amazon finds under $30 that can make everything in your home look new",
    "Your home office needs a good printer. So we found 3 of the best options out there",
    "24 life-changing under $25 products that actually work",
    "Expert-backed guides",
    "Underscored New Balance 327 Sneakers",
    "The best New Balance walking shoes, according to podiatrists",
    "The best products on sale this weekend that are truly discounted",
    "Yes, you need a proper office chair. We tested 17 models to find the best one",
    "We tested 31 bidets over 4 years, but only 3 are worth your money",
    "16 anti-theft travel bags that will keep your belongings safe and give you peace of mind",
    "Editors' picks",
    "frigidaire window ac in window cnnu.jpg",
    "No central air? These window air conditioners are the next best thing",
    "REI is clearing out winter jackets from Patagonia, Columbia, Arc’teryx and more",
    "Need relief for flat feet? These are 12 of the best walking shoes recommended by podiatrists",
    "The best coolers in 2025, tested by editors",
    "9 of the best products released this week"
]




Trfm = AutoTextCleaner(
    universal_sep=' ',
    case_sensitive=True,
    global_flags=None,
    remove_empty_rows=True,
    join_2D=' ',
    return_dim=None,
    strip=True,
    replace=((re.compile('[^a-zA-Z0-9]'), ''), ),
    remove=(''),
    normalize=True,
    lexicon_lookup='auto_delete',
    remove_stops=False,
    ngram_merge=None,
    justify=79,
    get_statistics={'before':True, 'after':True}
)


out = Trfm.transform(text)

for line in out:
    print(line)









