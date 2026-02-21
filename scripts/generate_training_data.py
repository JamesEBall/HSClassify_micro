"""
Generate synthetic training data for HS code classification.

Creates realistic product descriptions in multiple languages mapped to HS codes.
Since we can't access external APIs, we synthesize diverse examples covering
major HS code chapters.
"""

import json
import csv
import random
import os

# HS Code structure: Chapter (2-digit) -> Heading (4-digit) -> Subheading (6-digit)
# We'll focus on common trade categories with 6-digit codes

HS_CODES = {
    # Chapter 01-05: Live Animals; Animal Products
    "020130": {"desc": "Fresh or chilled boneless bovine meat", "chapter": "Meat"},
    "020230": {"desc": "Frozen boneless bovine meat", "chapter": "Meat"},
    "030389": {"desc": "Other frozen fish", "chapter": "Fish"},
    "030617": {"desc": "Frozen shrimps and prawns", "chapter": "Fish"},
    "040120": {"desc": "Milk with fat content 1-6%", "chapter": "Dairy"},
    "040210": {"desc": "Milk powder, fat content ≤1.5%", "chapter": "Dairy"},
    "040690": {"desc": "Other cheese", "chapter": "Dairy"},
    
    # Chapter 06-14: Vegetable Products
    "060110": {"desc": "Bulbs, tubers for planting", "chapter": "Plants"},
    "070200": {"desc": "Tomatoes, fresh or chilled", "chapter": "Vegetables"},
    "070310": {"desc": "Fresh or chilled onions and shallots", "chapter": "Vegetables"},
    "070820": {"desc": "Beans, fresh or chilled", "chapter": "Vegetables"},
    "080810": {"desc": "Fresh apples", "chapter": "Fruits"},
    "080300": {"desc": "Bananas, fresh or dried", "chapter": "Fruits"},
    "080510": {"desc": "Fresh oranges", "chapter": "Fruits"},
    "090111": {"desc": "Coffee, not roasted, not decaffeinated", "chapter": "Coffee/Tea"},
    "090210": {"desc": "Green tea in packages ≤3kg", "chapter": "Coffee/Tea"},
    "090411": {"desc": "Pepper, neither crushed nor ground", "chapter": "Spices"},
    "100199": {"desc": "Wheat and meslin, other", "chapter": "Cereals"},
    "100630": {"desc": "Semi-milled or wholly milled rice", "chapter": "Cereals"},
    "120190": {"desc": "Soya beans, other", "chapter": "Oil Seeds"},
    "130219": {"desc": "Vegetable saps and extracts", "chapter": "Plants"},
    
    # Chapter 15-24: Food Products
    "150710": {"desc": "Crude soya-bean oil", "chapter": "Oils"},
    "151190": {"desc": "Other palm oil", "chapter": "Oils"},
    "170199": {"desc": "Other cane or beet sugar", "chapter": "Sugar"},
    "180100": {"desc": "Cocoa beans, whole or broken", "chapter": "Cocoa"},
    "190531": {"desc": "Sweet biscuits", "chapter": "Food Preparations"},
    "200990": {"desc": "Mixtures of fruit juices", "chapter": "Food Preparations"},
    "210111": {"desc": "Extracts and concentrates of coffee", "chapter": "Food Preparations"},
    "220210": {"desc": "Waters with added sugar or flavour", "chapter": "Beverages"},
    "220300": {"desc": "Beer made from malt", "chapter": "Beverages"},
    "220421": {"desc": "Wine in containers ≤2L", "chapter": "Beverages"},
    "220830": {"desc": "Whiskies", "chapter": "Beverages"},
    "240120": {"desc": "Tobacco, partly or wholly stemmed", "chapter": "Tobacco"},
    
    # Chapter 25-27: Mineral Products
    "252329": {"desc": "Portland cement, other", "chapter": "Minerals"},
    "270900": {"desc": "Petroleum oils, crude", "chapter": "Mineral Fuels"},
    "271012": {"desc": "Light petroleum oils", "chapter": "Mineral Fuels"},
    "271019": {"desc": "Other medium petroleum oils", "chapter": "Mineral Fuels"},
    "271111": {"desc": "Natural gas, liquefied", "chapter": "Mineral Fuels"},
    "271600": {"desc": "Electrical energy", "chapter": "Mineral Fuels"},
    
    # Chapter 28-38: Chemical Products
    "280461": {"desc": "Silicon, containing ≥99.99% Si", "chapter": "Chemicals"},
    "290531": {"desc": "Ethylene glycol", "chapter": "Chemicals"},
    "300490": {"desc": "Other medicaments, packaged for retail", "chapter": "Pharmaceuticals"},
    "300220": {"desc": "Vaccines for human medicine", "chapter": "Pharmaceuticals"},
    "310520": {"desc": "Mineral or chemical fertilizers with NPK", "chapter": "Chemicals"},
    "330499": {"desc": "Other beauty or make-up preparations", "chapter": "Cosmetics"},
    "340111": {"desc": "Toilet soap in bars", "chapter": "Cosmetics"},
    "380891": {"desc": "Insecticides", "chapter": "Chemicals"},
    
    # Chapter 39-40: Plastics and Rubber
    "390110": {"desc": "Polyethylene, specific gravity <0.94", "chapter": "Plastics"},
    "390120": {"desc": "Polyethylene, specific gravity ≥0.94", "chapter": "Plastics"},
    "390760": {"desc": "Polyethylene terephthalate (PET)", "chapter": "Plastics"},
    "392010": {"desc": "Plates and sheets of ethylene polymers", "chapter": "Plastics"},
    "392321": {"desc": "Sacks and bags of ethylene polymers", "chapter": "Plastics"},
    "401110": {"desc": "New pneumatic tyres for cars", "chapter": "Rubber"},
    "401120": {"desc": "New pneumatic tyres for buses/lorries", "chapter": "Rubber"},
    
    # Chapter 44-49: Wood, Paper
    "440710": {"desc": "Lumber, coniferous, sawn", "chapter": "Wood"},
    "470321": {"desc": "Chemical wood pulp, bleached", "chapter": "Paper"},
    "480256": {"desc": "Uncoated paper, 40-150 g/m²", "chapter": "Paper"},
    "481910": {"desc": "Cartons, boxes of corrugated paper", "chapter": "Paper"},
    
    # Chapter 50-63: Textiles
    "520100": {"desc": "Cotton, not carded or combed", "chapter": "Textiles"},
    "520812": {"desc": "Unbleached plain weave cotton", "chapter": "Textiles"},
    "540233": {"desc": "Textured polyester yarn", "chapter": "Textiles"},
    "610910": {"desc": "T-shirts and singlets, cotton, knitted", "chapter": "Garments"},
    "611030": {"desc": "Jerseys, pullovers of man-made fibres", "chapter": "Garments"},
    "620342": {"desc": "Men's trousers of cotton", "chapter": "Garments"},
    "620462": {"desc": "Women's trousers of cotton", "chapter": "Garments"},
    "630260": {"desc": "Toilet and kitchen linen of cotton", "chapter": "Textiles"},
    "640399": {"desc": "Other footwear with rubber soles", "chapter": "Footwear"},
    "640510": {"desc": "Footwear with uppers of leather", "chapter": "Footwear"},
    
    # Chapter 72-83: Base Metals
    "720839": {"desc": "Hot-rolled iron or steel, coil >600mm", "chapter": "Steel"},
    "720917": {"desc": "Cold-rolled iron or steel coil", "chapter": "Steel"},
    "730890": {"desc": "Other structures of iron or steel", "chapter": "Steel"},
    "740311": {"desc": "Refined copper cathodes", "chapter": "Metals"},
    "760110": {"desc": "Unwrought aluminium, not alloyed", "chapter": "Metals"},
    "760120": {"desc": "Unwrought aluminium alloys", "chapter": "Metals"},
    
    # Chapter 84: Machinery
    "840734": {"desc": "Spark-ignition engines >1000cc", "chapter": "Machinery"},
    "841191": {"desc": "Parts of turbo-jets", "chapter": "Machinery"},
    "841810": {"desc": "Combined refrigerator-freezers", "chapter": "Machinery"},
    "841821": {"desc": "Household refrigerators, compression", "chapter": "Machinery"},
    "841911": {"desc": "Instantaneous gas water heaters", "chapter": "Machinery"},
    "842139": {"desc": "Other filtering or purifying machinery", "chapter": "Machinery"},
    "842199": {"desc": "Parts for filtering machinery", "chapter": "Machinery"},
    "843149": {"desc": "Other parts of cranes", "chapter": "Machinery"},
    "847130": {"desc": "Portable digital computers <10kg", "chapter": "Electronics"},
    "847141": {"desc": "Other digital processing units", "chapter": "Electronics"},
    "847150": {"desc": "Processing units for digital machines", "chapter": "Electronics"},
    "847170": {"desc": "Storage units for computers", "chapter": "Electronics"},
    "847330": {"desc": "Parts and accessories for computers", "chapter": "Electronics"},
    
    # Chapter 85: Electrical/Electronics
    "850440": {"desc": "Static converters", "chapter": "Electronics"},
    "850710": {"desc": "Lead-acid accumulators for vehicles", "chapter": "Electronics"},
    "850760": {"desc": "Lithium-ion accumulators", "chapter": "Electronics"},
    "851712": {"desc": "Smartphones", "chapter": "Electronics"},
    "852580": {"desc": "Television cameras and digital cameras", "chapter": "Electronics"},
    "852872": {"desc": "Other colour TV receivers", "chapter": "Electronics"},
    "853400": {"desc": "Printed circuits", "chapter": "Electronics"},
    "854140": {"desc": "Photosensitive semiconductor devices", "chapter": "Electronics"},
    "854231": {"desc": "Electronic integrated circuits, processors", "chapter": "Electronics"},
    "854232": {"desc": "Electronic integrated circuits, memories", "chapter": "Electronics"},
    "854239": {"desc": "Other electronic integrated circuits", "chapter": "Electronics"},
    
    # Chapter 87: Vehicles
    "870120": {"desc": "Road tractors for semi-trailers", "chapter": "Vehicles"},
    "870323": {"desc": "Motor cars, 1500-3000cc", "chapter": "Vehicles"},
    "870324": {"desc": "Motor cars, >3000cc", "chapter": "Vehicles"},
    "870332": {"desc": "Diesel motor cars, 1500-2500cc", "chapter": "Vehicles"},
    "870340": {"desc": "Electric motor cars (hybrid)", "chapter": "Vehicles"},
    "870380": {"desc": "Other motor vehicles, electric", "chapter": "Vehicles"},
    "870899": {"desc": "Other parts for motor vehicles", "chapter": "Vehicles"},
    "871120": {"desc": "Motorcycles, 50-250cc", "chapter": "Vehicles"},
    
    # Chapter 88-89: Aircraft, Ships
    "880230": {"desc": "Aeroplanes, unladen weight 2000-15000kg", "chapter": "Aircraft"},
    "890190": {"desc": "Other cargo vessels", "chapter": "Ships"},
    
    # Chapter 90: Instruments
    "900190": {"desc": "Optical fibres and bundles", "chapter": "Instruments"},
    "901839": {"desc": "Other needles, catheters", "chapter": "Medical"},
    "901890": {"desc": "Other medical instruments", "chapter": "Medical"},
    "902620": {"desc": "Instruments for measuring pressure", "chapter": "Instruments"},
    
    # Chapter 94-96: Furniture, Misc.
    "940161": {"desc": "Upholstered seats with wooden frames", "chapter": "Furniture"},
    "940180": {"desc": "Other seats", "chapter": "Furniture"},
    "940350": {"desc": "Wooden bedroom furniture", "chapter": "Furniture"},
    "940360": {"desc": "Other wooden furniture", "chapter": "Furniture"},
    "950300": {"desc": "Tricycles, scooters and pedal cars (toys)", "chapter": "Toys"},
    "950490": {"desc": "Other games and articles for games", "chapter": "Toys"},
}

# Product description templates in multiple languages
# English variations for each HS code
ENGLISH_TEMPLATES = {
    "020130": [
        "Fresh boneless beef cuts for restaurant supply",
        "Chilled boneless bovine meat, premium grade",
        "Boneless beef sirloin, fresh, not frozen",
        "Premium quality fresh deboned cattle meat",
        "Chilled beef tenderloin, boneless, vacuum packed",
        "Fresh boneless cow meat for wholesale distribution",
        "Grade A boneless beef, refrigerated",
        "Deboned fresh bovine meat for retail",
        "Fresh beef brisket, boneless, chilled",
        "Chilled boneless veal, premium cuts",
    ],
    "020230": [
        "Frozen boneless beef for export",
        "Boneless frozen bovine meat, bulk pack",
        "Frozen deboned cattle meat, commercial grade",
        "IQF boneless beef cuts, frozen at -18°C",
        "Frozen premium boneless beef quarters",
        "Bulk frozen boneless bovine meat shipment",
        "Frozen boneless stewing beef, 20kg cartons",
        "Deep frozen deboned beef for processing",
        "Frozen boneless beef trimmings, 80/20",
        "Boneless frozen cow meat, HALAL certified",
    ],
    "030617": [
        "Frozen shrimp, headless, shell-on",
        "Frozen prawns for seafood restaurant",
        "IQF frozen vannamei shrimp, 31/40 count",
        "Frozen black tiger prawns, peeled and deveined",
        "Bulk frozen shrimp, 10kg master carton",
        "Frozen cooked prawns, tail-on",
        "Frozen raw shrimp, head-on shell-on",
        "Frozen jumbo prawns, 16/20 count",
        "Frozen breaded shrimp for food service",
        "Block frozen whole shrimp, commercial grade",
    ],
    "070200": [
        "Fresh tomatoes from greenhouse production",
        "Cherry tomatoes, fresh, in 250g punnets",
        "Roma tomatoes, fresh, for canning factory",
        "Fresh vine-ripened tomatoes, 5kg boxes",
        "Organic fresh tomatoes, hydroponic",
        "Fresh beefsteak tomatoes for supermarket",
        "Heirloom fresh tomatoes, mixed varieties",
        "Fresh plum tomatoes for Italian cooking",
        "Greenhouse tomatoes, fresh, grade A",
        "Fresh cocktail tomatoes, premium pack",
    ],
    "080810": [
        "Fresh Fuji apples from Washington State",
        "Fresh Gala apples, premium grade, 18kg boxes",
        "Red Delicious apples, fresh, for export",
        "Fresh organic Granny Smith apples",
        "Fresh Honeycrisp apples, sorted by size",
        "Green apples, fresh, retail ready packaging",
        "Fresh Pink Lady apples, 80 count boxes",
        "Golden Delicious fresh apples, class I",
        "Fresh Braeburn apples, orchard packed",
        "Royal Gala fresh apples for supermarket",
    ],
    "090111": [
        "Green coffee beans, Arabica, unwashed",
        "Unroasted coffee beans, Colombian origin",
        "Raw green coffee, robusta, not decaffeinated",
        "Coffee beans, not roasted, in 60kg jute bags",
        "Green Arabica coffee, single origin Ethiopia",
        "Unprocessed coffee beans, Vietnam robusta",
        "Raw coffee beans, Brazilian Santos grade",
        "Green coffee, not roasted, specialty grade",
        "Unroasted decaf-free coffee, 1000kg lot",
        "Green coffee beans, Indonesian Sumatra",
    ],
    "100630": [
        "Thai jasmine rice, white, milled, 5% broken",
        "Basmati rice, fully milled, 25kg bags",
        "Vietnamese broken rice, wholly milled",
        "Long grain white rice, polished, bulk",
        "Japanese sushi rice, short grain, milled",
        "Semi-milled brown rice, organic",
        "Premium jasmine rice, 100% grade A",
        "Milled parboiled rice, fortified",
        "Sticky rice, glutinous, wholly milled",
        "Fragrant rice, white, fully polished",
    ],
    "170199": [
        "White refined cane sugar, ICUMSA 45",
        "Raw brown sugar, unrefined, bulk",
        "Granulated beet sugar, retail packs",
        "Cane sugar, crystal white, 50kg bags",
        "Sugar, refined, in 1kg consumer packs",
        "Industrial grade white sugar for beverages",
        "Organic raw cane sugar, fair trade",
        "Turbinado sugar, partially refined",
        "Fine granulated white sugar, food grade",
        "Demerara sugar, unrefined cane sugar",
    ],
    "220300": [
        "Craft beer, IPA style, 330ml bottles",
        "Lager beer, malt based, 500ml cans",
        "Stout beer, premium, 6-pack bottles",
        "Pale ale, microbrewery, kegs 50L",
        "Wheat beer, unfiltered, 12-pack",
        "Pilsner beer, imported, 24x330ml case",
        "Dark beer from malt, bottled, 7% ABV",
        "Light beer, low calorie, 355ml cans",
        "Amber ale, artisan craft beer",
        "Non-alcoholic malt beer, 0.5% ABV",
    ],
    "271012": [
        "Light petroleum distillates, gasoline grade",
        "Motor spirit (petrol), RON 95",
        "Aviation gasoline, AVGAS 100LL",
        "Naphtha, light fraction, for petrochemicals",
        "Gasoline, unleaded, for motor vehicles",
        "Light petroleum oil, for blending",
        "Premium unleaded petrol, 98 octane",
        "Straight run gasoline, refinery output",
        "Reformate, high octane, light petroleum",
        "Light distillate feedstock for crackers",
    ],
    "300490": [
        "Paracetamol tablets, 500mg, retail packed",
        "Ibuprofen capsules, 200mg, blister pack",
        "Amoxicillin oral suspension, 125mg/5ml",
        "Metformin tablets for diabetes, 500mg",
        "Omeprazole capsules, 20mg, 30 count",
        "Blood pressure medication, packaged, retail",
        "Antihistamine tablets, cetirizine 10mg",
        "Multivitamin supplement tablets, 60 count",
        "Cough syrup, retail packaged, 200ml bottle",
        "Pain relief gel, diclofenac 1%, 50g tube",
    ],
    "390120": [
        "High density polyethylene resin, pellets",
        "HDPE granules for blow moulding",
        "Polyethylene, high density, film grade",
        "HDPE pipe grade resin, black compound",
        "High density PE pellets, injection grade",
        "HDPE resin, specific gravity >0.94",
        "Polyethylene granules for plastic bags",
        "HDPE virgin resin, natural colour",
        "High density polyethylene for containers",
        "PE-HD compound, UV stabilized, pellets",
    ],
    "392321": [
        "Plastic carrier bags, polyethylene",
        "HDPE garbage bags, black, 50L capacity",
        "Shopping bags of ethylene polymers",
        "Clear poly bags for food packaging",
        "Resealable plastic bags, LDPE",
        "Industrial polyethylene sacks, heavy duty",
        "Zip-lock bags, polyethylene, various sizes",
        "Produce bags, HDPE, on rolls",
        "T-shirt style carry bags, plastic",
        "Poly mailer bags for shipping",
    ],
    "401110": [
        "New radial tyres for passenger cars, 205/55R16",
        "Car tires, new, all-season, size 225/45R17",
        "Pneumatic rubber tires for sedan, new",
        "New summer tyres for automobiles, 195/65R15",
        "Performance car tires, new, 245/40R18",
        "New winter snow tires for cars",
        "Economy car tyres, new, 175/70R13",
        "SUV tyres, new, 265/70R16, highway terrain",
        "New all-terrain tyres for passenger vehicles",
        "Run-flat tires for cars, new, 225/50R17",
    ],
    "520100": [
        "Raw cotton bales, not carded or combed",
        "Upland cotton, machine picked, bales",
        "Egyptian long staple cotton, raw",
        "Cotton lint, ginned, in standard bales",
        "Pima cotton, extra long staple, raw",
        "Organic raw cotton, not processed",
        "Raw cotton fibre, medium staple",
        "American upland cotton, uncarded",
        "Cotton, raw, compressed bales for spinning",
        "Uzbek cotton, uncarded, class A",
    ],
    "610910": [
        "Cotton T-shirts, men's, knitted, crew neck",
        "Women's cotton T-shirt, printed, knitted",
        "Children's T-shirts, 100% cotton, knitted",
        "Plain white cotton undershirts, pack of 3",
        "Graphic print T-shirts, cotton knit",
        "Organic cotton T-shirts, unisex, knitted",
        "Sports T-shirts, cotton blend, knitted",
        "Polo shirts, cotton knit, short sleeve",
        "Basic cotton singlets, knitted, multipack",
        "Oversized T-shirts, cotton jersey knit",
    ],
    "620342": [
        "Men's cotton trousers, woven, casual",
        "Men's denim jeans, blue, cotton",
        "Cotton chino pants for men, beige",
        "Men's cargo trousers, cotton twill",
        "Formal cotton trousers, men's, tailored",
        "Men's workwear pants, 100% cotton",
        "Cotton corduroy trousers, men's",
        "Men's slim fit cotton jeans",
        "Khaki cotton pants, men's casual",
        "Men's cotton shorts, woven, knee length",
    ],
    "640399": [
        "Rubber sole sneakers, synthetic upper",
        "Sport shoes with rubber outsole",
        "Casual trainers, rubber sole, canvas upper",
        "Running shoes, rubber bottom, mesh",
        "Slip-on shoes, rubber sole, fabric upper",
        "Canvas shoes with vulcanized rubber sole",
        "Sandals with rubber soles",
        "Rubber sole work boots, synthetic",
        "Athletic shoes, rubber outsole, knit upper",
        "Children's shoes, rubber soles, textile",
    ],
    "720839": [
        "Hot rolled steel coil, width >600mm, 3mm thick",
        "HRC steel sheet in coils, Q235B grade",
        "Hot rolled mild steel coil, pickled",
        "Steel coils, hot rolled, 1250mm width",
        "HR steel plate in coils, 6mm thickness",
        "Hot rolled carbon steel strip, wide",
        "HRC coil, structural grade, 4.5mm",
        "Hot rolled steel sheets, S235JR, coiled",
        "Flat rolled iron coil, hot rolled, >600mm",
        "HR steel coil for pipe manufacturing",
    ],
    "760110": [
        "Primary aluminium ingots, 99.7% pure",
        "Unwrought aluminium billets, unalloyed",
        "Aluminium T-bars, primary, not alloyed",
        "Pure aluminium sow ingots, 25kg each",
        "Unwrought aluminium slabs, 99.5% Al",
        "Primary aluminium wire bar, unalloyed",
        "Aluminium pigs, high purity, unwrought",
        "P1020 aluminium ingots, unalloyed",
        "Unwrought aluminium, electrolytic grade",
        "Primary aluminium, 99.85% pure, ingots",
    ],
    "847130": [
        "Laptop computer, 14 inch, 16GB RAM",
        "Portable notebook PC, lightweight, i7 processor",
        "MacBook Pro, 13 inch, M2 chip",
        "Ultrabook laptop, under 10kg, for business",
        "Student laptop, 15.6 inch, budget model",
        "2-in-1 convertible laptop tablet",
        "Gaming laptop, portable, 17 inch screen",
        "Chromebook, portable digital computer",
        "Thin and light laptop, OLED display",
        "Refurbished portable computer, 14 inch",
    ],
    "851712": [
        "Apple iPhone 15, 256GB, unlocked",
        "Samsung Galaxy S24, 5G smartphone",
        "Xiaomi Redmi Note 13, dual SIM phone",
        "Google Pixel 8 smartphone, 128GB",
        "OnePlus 12 mobile phone, 5G capable",
        "Smartphone, Android, 6.7 inch display",
        "Budget smartphone, 4G LTE, 64GB",
        "Refurbished iPhone 14, 128GB",
        "Huawei mobile phone, dual camera",
        "5G smartphone, OLED screen, 512GB storage",
    ],
    "852872": [
        "55 inch LED television, 4K UHD",
        "Smart TV, 65 inch, colour, with WiFi",
        "OLED television set, 77 inch, colour",
        "Colour LCD TV, 43 inch, HD ready",
        "Samsung 50 inch QLED colour TV",
        "Android smart television, 32 inch colour",
        "Large screen colour TV, 85 inch, 8K",
        "Portable colour television, 24 inch",
        "Curved OLED colour TV, 55 inch",
        "4K Ultra HD colour television receiver",
    ],
    "854231": [
        "Microprocessor chips, 7nm, for computers",
        "CPU integrated circuits, server grade",
        "ARM-based processor ICs, mobile SoC",
        "Digital signal processors, IC chips",
        "GPU chips, graphics processing units",
        "FPGA chips, programmable logic ICs",
        "Microcontroller ICs, 32-bit, embedded",
        "Application processor chips for smartphones",
        "AI accelerator chips, neural processing",
        "RISC-V processor integrated circuits",
    ],
    "850760": [
        "Lithium-ion battery cells, 18650 type",
        "Li-ion battery pack for electric vehicles",
        "Rechargeable lithium-ion batteries, 3.7V",
        "Lithium polymer battery, for smartphones",
        "EV battery modules, lithium-ion, 400V",
        "Li-ion accumulator, cylindrical, 21700",
        "Lithium iron phosphate batteries, LFP",
        "Lithium-ion power bank, 20000mAh",
        "Battery cells, lithium-ion, prismatic",
        "NMC lithium-ion batteries for storage",
    ],
    "870323": [
        "Sedan car, petrol engine, 2000cc",
        "Toyota Camry, gasoline, 2.5L engine",
        "Honda Civic, spark ignition, 1800cc",
        "Passenger automobile, 1500-3000cc petrol",
        "SUV, gasoline engine, 2400cc",
        "Compact car, petrol, 1600cc, new",
        "Family sedan, 2.0L gasoline engine",
        "Sports car, petrol, 2500cc turbo",
        "Station wagon, spark ignition, 2000cc",
        "New passenger car, petrol 1.5-3.0L",
    ],
    "870380": [
        "Electric passenger car, battery powered",
        "Tesla Model 3, pure electric vehicle",
        "BYD electric car, long range battery",
        "Electric SUV, 400km range, BEV",
        "Compact electric vehicle for city driving",
        "Electric automobile, lithium-ion powered",
        "Fully electric passenger car, 75kWh battery",
        "EV sedan, electric motor, zero emissions",
        "Battery electric vehicle, new, for passengers",
        "Electric car, dual motor, all-wheel drive",
    ],
    "940350": [
        "Wooden bedroom set, bed frame and nightstands",
        "Oak wood wardrobe for bedroom",
        "Solid pine chest of drawers, bedroom",
        "Wooden bed frame, queen size, teak",
        "Bedroom vanity table, mahogany wood",
        "Wooden headboard, upholstered, for bed",
        "Children's bedroom furniture set, wood",
        "Wooden nightstand with two drawers",
        "Bedroom armoire, solid wood, walnut",
        "Platform bed, wooden, king size, modern",
    ],
    "950300": [
        "Children's tricycle, pedal operated",
        "Toy scooter for kids, push type",
        "Pedal car for children, plastic body",
        "Kids' balance bicycle, toy grade",
        "Toy ride-on car, battery operated",
        "Children's go-kart, pedal powered",
        "Miniature electric car for toddlers",
        "Toy pedal tractor for children",
        "Push scooter for kids, 3 wheels",
        "Children's toy wagon, pull-along",
    ],
}

# ============================================================
# Expanded multilingual templates covering all 118 curated codes
# Target: ~5 examples per language per code for TH/VI/ZH
# ============================================================

THAI_TEMPLATES = {
    # Meat
    "020130": ["เนื้อวัวสด ไม่มีกระดูก สำหรับร้านอาหาร", "เนื้อโคสดแช่เย็น ปลอดกระดูก คุณภาพดี", "เนื้อวัวไม่มีกระดูก สด ไม่แช่แข็ง", "เนื้อโคชิ้นพิเศษ สด ไม่ติดกระดูก", "เนื้อสันในวัวสด ปลอดกระดูก บรรจุสูญญากาศ"],
    "020230": ["เนื้อวัวแช่แข็ง ไม่มีกระดูก สำหรับส่งออก", "เนื้อโคแช่แข็งปลอดกระดูก บรรจุกล่อง 20 กก.", "เนื้อวัวบดแช่แข็ง สำหรับแปรรูป", "เนื้อแช่แข็งไร้กระดูก ฮาลาล", "เนื้อโคแช่แข็ง ตัดแต่งพร้อมจำหน่าย"],
    # Fish
    "030389": ["ปลาแช่แข็งชนิดอื่น สำหรับส่งออก", "ปลาทะเลแช่แข็ง ทั้งตัว", "ปลาน้ำจืดแช่แข็ง บรรจุกล่อง", "ปลาสดแช่แข็ง สำหรับอุตสาหกรรมอาหาร", "ปลาแช่แข็งเกรดพาณิชย์"],
    "030617": ["กุ้งแช่แข็ง แกะหัว ติดเปลือก", "กุ้งขาวแวนนาไมแช่แข็ง ขนาด 31/40", "กุ้งกุลาดำแช่แข็ง ปอกเปลือก", "กุ้งสดแช่แข็ง สำหรับร้านอาหาร", "กุ้งแช่แข็ง IQF ส่งออก"],
    # Dairy
    "040120": ["นมพร่องมันเนย ไขมัน 1-6%", "นมสดพาสเจอไรซ์ ไขมันต่ำ", "นม UHT ไขมัน 3.2%", "นมวัวสด ไม่ปรุงแต่ง", "นมพร้อมดื่ม ไขมันปานกลาง"],
    "040210": ["นมผง ไขมันต่ำ ไม่เกิน 1.5%", "นมผงขาดมันเนย สำหรับอุตสาหกรรม", "นมผงสำหรับเด็ก ไขมันต่ำ", "นมผงพร่องมันเนย บรรจุถุง 25 กก."],
    "040690": ["เนยแข็งชนิดอื่น สำหรับปรุงอาหาร", "ชีสนำเข้าจากยุโรป", "เนยแข็งเชดดาร์ บรรจุแพ็ค", "ชีสมอสซาเรลลา สำหรับพิซซ่า"],
    # Vegetables
    "070200": ["มะเขือเทศสด จากโรงเรือน", "มะเขือเทศเชอร์รี่สด แพ็ค 250 กรัม", "มะเขือเทศสดคุณภาพดี สำหรับส่งออก", "มะเขือเทศสดเกรด A ปลูกในโรงเรือน", "มะเขือเทศราชินีสด บรรจุกล่อง"],
    "070310": ["หัวหอมสด แช่เย็น", "หอมแดงสด สำหรับส่งออก", "หอมหัวใหญ่สด เกรดพรีเมียม", "หัวหอมสดจากไร่ บรรจุถุง 10 กก."],
    "070820": ["ถั่วฝักยาวสด แช่เย็น", "ถั่วเขียวสด เก็บเกี่ยวใหม่", "ถั่วแขกสด สำหรับส่งออก", "ถั่วลันเตาสด คุณภาพดี"],
    # Fruits
    "080810": ["แอปเปิ้ลสด นำเข้า เกรดพรีเมียม", "แอปเปิ้ลฟูจิสด บรรจุกล่อง 18 กก.", "แอปเปิ้ลกาล่าสด จากนิวซีแลนด์", "แอปเปิ้ลเขียวสด แกรนนี่สมิธ", "แอปเปิ้ลสดคุณภาพส่งออก"],
    "080300": ["กล้วยหอมสด สำหรับส่งออก", "กล้วยสดจากสวน พร้อมจำหน่าย", "กล้วยไข่สด คุณภาพเกรด A", "กล้วยหอมทองสด บรรจุกล่อง"],
    "080510": ["ส้มสด เกรดพรีเมียม", "ส้มเขียวหวานสด จากสวน", "ส้มแมนดารินสด บรรจุกล่อง", "ส้มนาเวลสดจากแคลิฟอร์เนีย"],
    # Coffee/Tea
    "090111": ["เมล็ดกาแฟดิบ อาราบิก้า ยังไม่คั่ว", "กาแฟสารดิบ โรบัสต้า ยังไม่ผ่านการคั่ว", "เมล็ดกาแฟเขียว ไม่ได้คั่ว ไม่ได้สกัดคาเฟอีน", "กาแฟดิบจากเอธิโอเปีย เกรดพิเศษ", "เมล็ดกาแฟดิบ บรรจุกระสอบ 60 กก."],
    "090210": ["ชาเขียวบรรจุซอง ไม่เกิน 3 กก.", "ชาเขียวญี่ปุ่น แบบใบ", "ชาเขียวออร์แกนิค บรรจุถุง", "ชาเขียวมัทฉะ บรรจุกระป๋อง"],
    "090411": ["พริกไทยเม็ด ไม่บด", "พริกไทยดำทั้งเม็ด สำหรับส่งออก", "พริกไทยขาวเม็ด เกรดพรีเมียม", "พริกไทยดำ ไม่ผ่านการบด"],
    # Cereals
    "100199": ["ข้าวสาลี ชนิดอื่น", "ข้าวสาลีนำเข้า สำหรับอุตสาหกรรมขนมปัง", "เมล็ดข้าวสาลี เกรดพาณิชย์", "ข้าวสาลีเมสลิน สำหรับแปรรูป"],
    "100630": ["ข้าวหอมมะลิไทย ขัดสี 5% หัก", "ข้าวขาวสารขัดมัน บรรจุถุง 25 กก.", "ข้าวเหนียว ขัดสีแล้ว พร้อมส่งออก", "ข้าวบาสมาติ ขัดสีทั้งเมล็ด คุณภาพดี", "ข้าวขาวเมล็ดยาว ขัดสี สำหรับส่งออก"],
    "120190": ["ถั่วเหลือง ชนิดอื่น สำหรับสกัดน้ำมัน", "เมล็ดถั่วเหลือง นำเข้า สำหรับอุตสาหกรรม", "ถั่วเหลืองเกรดอาหารสัตว์"],
    # Sugar/Cocoa/Food
    "170199": ["น้ำตาลทรายขาวบริสุทธิ์ ICUMSA 45", "น้ำตาลทรายดิบ สีน้ำตาล จากอ้อย", "น้ำตาลทรายขาว บรรจุถุง 50 กก.", "น้ำตาลทรายแดง ไม่ผ่านกระบวนการฟอก", "น้ำตาลจากอ้อย เกรดอุตสาหกรรม"],
    "180100": ["เมล็ดโกโก้ดิบ ทั้งเม็ด", "เมล็ดคาเคาดิบ สำหรับแปรรูป", "โกโก้บีนส์ เกรดส่งออก"],
    "190531": ["ขนมปังกรอบหวาน บิสกิต", "คุกกี้บิสกิตหวาน บรรจุกล่อง", "ขนมปังกรอบ ชนิดหวาน นำเข้า"],
    "200990": ["น้ำผลไม้รวม บรรจุกล่อง", "น้ำผลไม้ผสม สำหรับขายปลีก", "น้ำผลไม้คั้นสด ชนิดรวม"],
    # Beverages
    "220210": ["น้ำดื่มผสมน้ำตาล หรือสารให้ความหวาน", "น้ำอัดลม บรรจุขวด", "น้ำแร่ผสมรสชาติ สำหรับขายปลีก"],
    "220300": ["เบียร์จากมอลต์ บรรจุขวด 330 มล.", "เบียร์ลาเกอร์ บรรจุกระป๋อง 500 มล.", "เบียร์คราฟต์ IPA สำหรับขายปลีก", "เบียร์สด จากมอลต์ บรรจุถัง 50 ลิตร"],
    "220421": ["ไวน์แดง บรรจุขวดไม่เกิน 2 ลิตร", "ไวน์ขาว จากฝรั่งเศส บรรจุขวด", "ไวน์องุ่น บรรจุภาชนะไม่เกิน 2 ลิตร"],
    "220830": ["วิสกี้ นำเข้าจากสก็อตแลนด์", "วิสกี้เบอร์เบิน จากอเมริกา", "วิสกี้ญี่ปุ่น บรรจุขวด"],
    "240120": ["ยาสูบ ลิดก้านแล้ว บางส่วนหรือทั้งหมด", "ใบยาสูบแปรรูป สำหรับอุตสาหกรรม"],
    # Minerals/Fuel
    "252329": ["ปูนซีเมนต์ปอร์ตแลนด์ ชนิดอื่น", "ปูนซีเมนต์สำหรับก่อสร้าง บรรจุถุง 50 กก."],
    "270900": ["น้ำมันดิบปิโตรเลียม", "น้ำมันดิบจากหลุมขุดเจาะ สำหรับกลั่น", "ปิโตรเลียมดิบ ส่งออกทางเรือ"],
    "271012": ["น้ำมันเบนซิน สำหรับรถยนต์", "น้ำมันปิโตรเลียมเบา เกรดเชื้อเพลิง", "น้ำมันเบนซินไร้สารตะกั่ว ออกเทน 95"],
    "271019": ["น้ำมันดีเซล สำหรับเครื่องยนต์", "น้ำมันปิโตรเลียมปานกลาง สำหรับอุตสาหกรรม"],
    "271111": ["ก๊าซธรรมชาติเหลว LNG", "ก๊าซ LNG สำหรับโรงไฟฟ้า", "ก๊าซธรรมชาติทำให้เป็นของเหลว ส่งทางเรือ"],
    "271600": ["พลังงานไฟฟ้า สำหรับส่งออก", "กระแสไฟฟ้า จากโรงไฟฟ้า"],
    # Chemicals
    "280461": ["ซิลิคอน ความบริสุทธิ์ 99.99%", "ซิลิคอนเกรดอิเล็กทรอนิกส์"],
    "290531": ["เอทิลีนไกลคอล สำหรับอุตสาหกรรม", "เอทิลีนไกลคอล เกรดอุตสาหกรรม"],
    "300490": ["ยารักษาโรค บรรจุสำหรับขายปลีก", "ยาแก้ปวดพาราเซตามอล 500 มก.", "ยาเม็ดลดไข้ บรรจุแผง", "ยาแก้อักเสบ บรรจุขวด"],
    "300220": ["วัคซีนสำหรับมนุษย์", "วัคซีนป้องกันโรคไข้หวัดใหญ่", "วัคซีนป้องกันโควิด สำหรับนำเข้า"],
    "310520": ["ปุ๋ยเคมี NPK สำหรับเกษตรกรรม", "ปุ๋ยสูตรผสม NPK บรรจุถุง"],
    "330499": ["เครื่องสำอาง ชนิดอื่น", "ผลิตภัณฑ์เสริมความงาม บรรจุขายปลีก"],
    "340111": ["สบู่ก้อน สำหรับอาบน้ำ", "สบู่ก้อนหอม สำหรับขายปลีก", "สบู่ก้อนออร์แกนิค"],
    "380891": ["ยาฆ่าแมลง สำหรับการเกษตร", "สารกำจัดแมลง บรรจุขวด"],
    # Plastics/Rubber
    "390110": ["เม็ดพลาสติกโพลีเอทิลีน ความหนาแน่นต่ำ LDPE", "LDPE เม็ดพลาสติก สำหรับผลิตฟิล์ม"],
    "390120": ["เม็ดพลาสติก HDPE ความหนาแน่นสูง", "HDPE เม็ดพลาสติก สำหรับขึ้นรูป", "โพลีเอทิลีนความหนาแน่นสูง เกรดท่อ"],
    "390760": ["เม็ดพลาสติก PET สำหรับผลิตขวด", "โพลีเอทิลีนเทเรฟทาเลต เกรดขวดน้ำ", "PET เรซิน บรรจุถุง 25 กก."],
    "392010": ["แผ่นพลาสติกโพลีเอทิลีน", "แผ่นฟิล์ม PE สำหรับบรรจุภัณฑ์"],
    "392321": ["ถุงพลาสติก PE สำหรับบรรจุสินค้า", "ถุงขยะ HDPE ขนาด 50 ลิตร", "ถุงหิ้วพลาสติก สำหรับซุปเปอร์มาร์เก็ต"],
    "401110": ["ยางรถยนต์ใหม่ ขนาด 205/55R16", "ยางรถเก๋งใหม่ สำหรับทุกฤดู", "ยางรถยนต์นั่งใหม่ แบบเรเดียล", "ยางรถ SUV ใหม่ ขนาด 265/70R16", "ยางรถยนต์ใหม่ สำหรับฤดูหนาว"],
    "401120": ["ยางรถบรรทุกใหม่ แบบเรเดียล", "ยางรถบัสใหม่ สำหรับงานหนัก", "ยางรถบรรทุกหนัก ขนาด 295/80R22.5"],
    # Wood/Paper
    "440710": ["ไม้สนแปรรูป เลื่อย สำหรับก่อสร้าง", "ไม้สนอัดแห้ง ขนาดมาตรฐาน"],
    "470321": ["เยื่อไม้เคมี ฟอกขาว", "เยื่อกระดาษฟอกขาว สำหรับผลิตกระดาษ"],
    "480256": ["กระดาษไม่เคลือบผิว น้ำหนัก 40-150 แกรม", "กระดาษพิมพ์เขียน A4 80 แกรม"],
    "481910": ["กล่องกระดาษลูกฟูก สำหรับบรรจุสินค้า", "กล่องลัง กระดาษลูกฟูก 5 ชั้น"],
    # Textiles/Garments
    "520100": ["ฝ้ายดิบ ยังไม่ผ่านการสาง", "ฝ้ายอัดก้อน สำหรับปั่นด้าย", "ฝ้ายดิบอินทรีย์ ยังไม่แปรรูป"],
    "520812": ["ผ้าฝ้ายทอธรรมดา ไม่ฟอก", "ผ้าดิบฝ้าย ทอเรียบ ยังไม่ฟอกขาว"],
    "540233": ["ด้ายโพลีเอสเตอร์เท็กเจอร์ สำหรับทอผ้า", "เส้นด้ายสังเคราะห์โพลีเอสเตอร์"],
    "610910": ["เสื้อยืดผ้าฝ้ายถัก คอกลม ผู้ชาย", "เสื้อยืดคอตตอน ผู้หญิง พิมพ์ลาย", "เสื้อยืดเด็ก ผ้าฝ้าย 100% ถักนิตติ้ง", "เสื้อยืดกีฬาผ้าฝ้ายถัก", "เสื้อโปโลผ้าฝ้ายถัก แขนสั้น"],
    "611030": ["เสื้อสเวตเตอร์ เส้นใยสังเคราะห์ ถัก", "เสื้อกันหนาว ไหมพรมสังเคราะห์"],
    "620342": ["กางเกงผู้ชาย ผ้าฝ้าย ทอ", "กางเกงยีนส์ชาย ผ้าฝ้าย สีน้ำเงิน", "กางเกงชิโน่ ผ้าฝ้าย สำหรับผู้ชาย"],
    "620462": ["กางเกงผู้หญิง ผ้าฝ้าย ทอ", "กางเกงสตรี ผ้าฝ้าย แบบสบาย"],
    "630260": ["ผ้าเช็ดตัว ผ้าเช็ดมือ ผ้าฝ้าย", "ผ้าเช็ดจาน ผ้าฝ้าย สำหรับครัว"],
    "640399": ["รองเท้าพื้นยาง หน้ารองเท้าสังเคราะห์", "รองเท้ากีฬาพื้นยาง ผ้าตาข่าย", "รองเท้าผ้าใบพื้นยาง สำหรับวิ่ง"],
    "640510": ["รองเท้าหนังแท้ สำหรับผู้ชาย", "รองเท้าหนัง เกรดพรีเมียม สำหรับทำงาน"],
    # Metals
    "720839": ["เหล็กแผ่นรีดร้อน ม้วน กว้างเกิน 600 มม.", "เหล็กม้วนรีดร้อน HRC เกรด Q235B", "แผ่นเหล็กกล้ารีดร้อน ความหนา 3 มม."],
    "720917": ["เหล็กแผ่นรีดเย็น ม้วน", "เหล็กม้วนรีดเย็น CRC สำหรับอุตสาหกรรม"],
    "730890": ["โครงสร้างเหล็กชนิดอื่น", "เหล็กโครงสร้าง สำหรับก่อสร้าง"],
    "740311": ["ทองแดงบริสุทธิ์ แผ่นแคโทด", "ทองแดงรีไฟน์ เกรดอิเล็กทรอนิกส์"],
    "760110": ["อะลูมิเนียมแท่ง ไม่ผสม ความบริสุทธิ์ 99.7%", "อะลูมิเนียมแท่งบิลเลท สำหรับรีด"],
    "760120": ["อะลูมิเนียมผสม แบบแท่ง", "อะลูมิเนียมอัลลอย สำหรับหล่อ"],
    # Machinery
    "840734": ["เครื่องยนต์จุดระเบิดด้วยประกายไฟ เกิน 1000 ซีซี", "เครื่องยนต์เบนซิน สำหรับรถยนต์"],
    "841191": ["ชิ้นส่วนของเครื่องยนต์เทอร์โบเจ็ท", "อะไหล่เครื่องยนต์ไอพ่น"],
    "841810": ["ตู้เย็นสองประตู แบบรวม", "ตู้เย็น-ช่องแช่แข็งรวม สำหรับใช้ในบ้าน"],
    "841821": ["ตู้เย็นในครัวเรือน แบบคอมเพรสเซอร์", "ตู้เย็นบ้าน ระบบอัดไอ ขนาด 300 ลิตร"],
    "841911": ["เครื่องทำน้ำร้อนแก๊ส แบบทันที", "เครื่องทำน้ำอุ่นแก๊ส สำหรับครัวเรือน"],
    "842139": ["เครื่องกรองหรือทำให้บริสุทธิ์ ชนิดอื่น", "เครื่องกรองอากาศอุตสาหกรรม"],
    "843149": ["ชิ้นส่วนของเครน ชนิดอื่น", "อะไหล่เครนและเครื่องจักรยก"],
    # Electronics
    "847130": ["คอมพิวเตอร์แล็ปท็อป จอ 14 นิ้ว", "โน้ตบุ๊คพกพา น้ำหนักเบา สำหรับทำงาน", "แล็ปท็อปสำหรับนักเรียน จอ 15.6 นิ้ว", "แล็ปท็อปเกมมิ่ง พกพา 17 นิ้ว", "แล็ปท็อป 2-in-1 พับจอสัมผัส"],
    "847141": ["หน่วยประมวลผลข้อมูลดิจิทัล ชนิดอื่น", "เซิร์ฟเวอร์คอมพิวเตอร์ สำหรับศูนย์ข้อมูล"],
    "847150": ["หน่วยประมวลผลสำหรับเครื่องคอมพิวเตอร์", "ซีพียู สำหรับเดสก์ท็อป"],
    "847170": ["หน่วยจัดเก็บข้อมูลคอมพิวเตอร์", "ฮาร์ดดิสก์ SSD 1TB", "อุปกรณ์จัดเก็บข้อมูล NVMe"],
    "847330": ["ชิ้นส่วนอุปกรณ์คอมพิวเตอร์", "อะไหล่เครื่องคอมพิวเตอร์ แรม คีย์บอร์ด"],
    "850440": ["อุปกรณ์แปลงไฟฟ้าแบบคงที่", "อินเวอร์เตอร์ สำหรับระบบโซลาร์เซลล์"],
    "850710": ["แบตเตอรี่ตะกั่ว-กรด สำหรับรถยนต์", "แบตเตอรี่รถยนต์ 12V 60Ah"],
    "850760": ["แบตเตอรี่ลิเธียม-ไอออน สำหรับรถ EV", "แบตเตอรี่ลิเธียมไอออน แบบชาร์จได้", "แบตเตอรี่ลิเธียมโพลิเมอร์ สำหรับมือถือ", "แบตเตอรี่ลิเธียม 18650 สำหรับอุตสาหกรรม", "แบตเตอรี่ LFP สำหรับระบบกักเก็บพลังงาน"],
    "851712": ["สมาร์ทโฟน แอนดรอยด์ จอ 6.7 นิ้ว", "โทรศัพท์มือถือ ไอโฟน 15 ความจุ 256GB", "มือถือ 5G ซัมซุง กาแล็กซี่ S24", "สมาร์ทโฟนราคาประหยัด 4G 64GB", "โทรศัพท์มือถือ 5G หน้าจอ OLED"],
    "852580": ["กล้องโทรทัศน์และกล้องดิจิทัล", "กล้องถ่ายวิดีโอ 4K สำหรับมืออาชีพ"],
    "852872": ["โทรทัศน์สี LED ขนาด 55 นิ้ว", "สมาร์ททีวี 65 นิ้ว 4K UHD", "ทีวีจอ OLED สี ขนาด 77 นิ้ว", "โทรทัศน์สี LCD 43 นิ้ว HD", "ทีวี QLED สี 50 นิ้ว สมาร์ททีวี"],
    "853400": ["แผงวงจรพิมพ์ PCB", "แผงวงจรสำหรับอุปกรณ์โทรคมนาคม"],
    "854140": ["อุปกรณ์สารกึ่งตัวนำไวแสง เซลล์แสงอาทิตย์", "แผงโซลาร์เซลล์ สำหรับผลิตไฟฟ้า"],
    "854231": ["ไมโครโปรเซสเซอร์ IC 7 นาโนเมตร", "ชิปซีพียู สำหรับเซิร์ฟเวอร์", "ชิป GPU สำหรับประมวลผลกราฟิก"],
    "854232": ["วงจรรวมหน่วยความจำ", "ชิปหน่วยความจำ DRAM 16GB"],
    "854239": ["วงจรรวมอิเล็กทรอนิกส์ ชนิดอื่น", "ชิป IC สำหรับระบบควบคุม"],
    # Vehicles
    "870120": ["รถลากจูงสำหรับกึ่งพ่วง", "หัวรถลาก สำหรับขนส่งสินค้า"],
    "870323": ["รถยนต์นั่งเครื่องเบนซิน 2000 ซีซี", "รถเก๋งสปาร์คอิกนิชั่น 1500-3000 ซีซี", "รถยนต์โตโยต้า คัมรี่ เครื่อง 2.5 ลิตร", "รถเก๋งเครื่องเบนซิน 1.6 ลิตร ใหม่", "รถ SUV เครื่องยนต์เบนซิน 2400 ซีซี"],
    "870324": ["รถยนต์นั่ง เครื่องยนต์เบนซินเกิน 3000 ซีซี", "รถสปอร์ต เครื่องยนต์เทอร์โบ เกิน 3 ลิตร"],
    "870332": ["รถยนต์ดีเซล 1500-2500 ซีซี", "รถกระบะดีเซล 2.4 ลิตร"],
    "870340": ["รถยนต์ไฮบริด ไฟฟ้า-เบนซิน", "รถยนต์ไฮบริด ปลั๊กอิน"],
    "870380": ["รถยนต์ไฟฟ้า พลังงานแบตเตอรี่", "รถ EV เทสลา โมเดล 3 ไฟฟ้าล้วน", "รถยนต์ไฟฟ้า BYD แบตเตอรี่ลิเธียม", "รถยนต์ไฟฟ้าขนาดกะทัดรัด สำหรับในเมือง", "รถ EV SUV วิ่งได้ 400 กม."],
    "870899": ["ชิ้นส่วนรถยนต์ ชนิดอื่น", "อะไหล่รถยนต์ นำเข้า"],
    "871120": ["รถจักรยานยนต์ 50-250 ซีซี", "มอเตอร์ไซค์ ขนาด 150 ซีซี"],
    # Aircraft/Ships
    "880230": ["เครื่องบิน น้ำหนักเปล่า 2000-15000 กก.", "เครื่องบินโดยสาร ขนาดกลาง"],
    "890190": ["เรือขนส่งสินค้า ชนิดอื่น", "เรือบรรทุกสินค้า สำหรับการค้าระหว่างประเทศ"],
    # Instruments/Medical
    "900190": ["เส้นใยนำแสง สำหรับโทรคมนาคม", "ไฟเบอร์ออปติก บรรจุมัด"],
    "901839": ["เข็มฉีดยา สายสวน ชนิดอื่น", "อุปกรณ์การแพทย์ สายสวนหลอดเลือด"],
    "901890": ["เครื่องมือแพทย์ ชนิดอื่น", "อุปกรณ์การแพทย์ สำหรับโรงพยาบาล"],
    "902620": ["เครื่องวัดความดัน สำหรับอุตสาหกรรม", "มาตรวัดแรงดัน เกจวัดความดัน"],
    # Furniture/Toys
    "940161": ["เก้าอี้บุนวม โครงไม้", "เก้าอี้รับแขก บุผ้า โครงไม้สัก"],
    "940180": ["เก้าอี้ชนิดอื่น", "เก้าอี้พลาสติก สำหรับสำนักงาน"],
    "940350": ["เฟอร์นิเจอร์ห้องนอน ไม้ เตียงและโต๊ะข้างเตียง", "ตู้เสื้อผ้าไม้โอ๊ค สำหรับห้องนอน", "เตียงไม้สัก ขนาดควีนไซส์"],
    "940360": ["เฟอร์นิเจอร์ไม้ ชนิดอื่น", "โต๊ะทำงานไม้ สำหรับสำนักงาน"],
    "950300": ["รถสามล้อเด็ก ถีบ", "สกู๊ตเตอร์ของเล่น สำหรับเด็ก", "รถไฟฟ้าเด็กเล่น"],
    "950490": ["เกมและของเล่นชนิดอื่น", "บอร์ดเกม สำหรับครอบครัว"],
    # Oils
    "150710": ["น้ำมันถั่วเหลืองดิบ สำหรับกลั่น", "น้ำมันถั่วเหลืองดิบ เกรดอุตสาหกรรม"],
    "151190": ["น้ำมันปาล์ม ชนิดอื่น", "น้ำมันปาล์มบริสุทธิ์ สำหรับปรุงอาหาร"],
    "130219": ["น้ำยางพืช สารสกัดจากพืช", "สารสกัดจากพืช สำหรับอุตสาหกรรม"],
    "060110": ["หัวพันธุ์ไม้ สำหรับปลูก", "หัวทิวลิป สำหรับเพาะปลูก"],
    "210111": ["กาแฟสำเร็จรูป สกัดเข้มข้น", "กาแฟผงสำเร็จรูป บรรจุถุง"],
    "842199": ["ชิ้นส่วนเครื่องกรอง ชนิดอื่น", "ไส้กรองอุตสาหกรรม"],
}

VIETNAMESE_TEMPLATES = {
    # Meat
    "020130": ["Thịt bò tươi không xương, cấp đông mát", "Thịt bò không xương tươi cho nhà hàng", "Thịt trâu bò tươi ướp lạnh, không có xương", "Thịt bò phi lê tươi, bỏ xương, đóng chân không", "Thịt bò tươi nguyên miếng, không xương, chất lượng cao"],
    "020230": ["Thịt bò đông lạnh không xương, xuất khẩu", "Thịt bò đông lạnh đã bỏ xương, thùng 20kg", "Thịt bò xay đông lạnh cho chế biến", "Thịt bò đông lạnh không xương, Halal"],
    # Fish
    "030389": ["Cá đông lạnh loại khác, xuất khẩu", "Cá biển đông lạnh nguyên con", "Cá nước ngọt đông lạnh, đóng thùng"],
    "030617": ["Tôm đông lạnh, bóc vỏ, bỏ đầu", "Tôm sú đông lạnh, nguyên con", "Tôm thẻ chân trắng đông lạnh xuất khẩu", "Tôm IQF đông lạnh 31/40 xuất khẩu", "Tôm đông lạnh, sơ chế, bỏ đầu bỏ vỏ"],
    # Dairy
    "040120": ["Sữa tươi, hàm lượng chất béo 1-6%", "Sữa tiệt trùng ít béo", "Sữa bò tươi nguyên chất"],
    "040210": ["Sữa bột, hàm lượng chất béo dưới 1.5%", "Sữa bột tách béo cho công nghiệp"],
    "040690": ["Phô mai loại khác", "Phô mai Cheddar nhập khẩu", "Phô mai Mozzarella cho pizza"],
    # Vegetables
    "070200": ["Cà chua tươi từ nhà kính", "Cà chua bi tươi, hộp 250g", "Cà chua tươi chất lượng cao xuất khẩu", "Cà chua tươi loại 1 trồng thủy canh"],
    "070310": ["Hành tây tươi, ướp lạnh", "Hành tím tươi xuất khẩu", "Hành tây lớn tươi, chất lượng cao"],
    "070820": ["Đậu tươi, ướp lạnh", "Đậu cô ve tươi cho xuất khẩu", "Đậu Hà Lan tươi, chất lượng tốt"],
    # Fruits
    "080810": ["Táo tươi nhập khẩu, loại premium", "Táo Fuji tươi, hộp 18kg", "Táo Gala tươi từ New Zealand", "Táo xanh Granny Smith tươi"],
    "080300": ["Chuối tươi xuất khẩu", "Chuối tươi từ vườn, sẵn sàng bán", "Chuối tiêu tươi, loại 1"],
    "080510": ["Cam tươi, loại premium", "Cam sành tươi từ vườn", "Cam navel tươi nhập khẩu"],
    # Coffee/Tea/Spices
    "090111": ["Cà phê nhân xanh, chưa rang, Robusta", "Hạt cà phê thô Arabica chưa rang", "Cà phê nguyên liệu, chưa qua chế biến", "Cà phê nhân xanh Việt Nam, Robusta, bao 60kg", "Hạt cà phê chưa rang, đặc sản, đơn vùng"],
    "090210": ["Trà xanh đóng gói dưới 3kg", "Trà xanh Nhật Bản, nguyên lá", "Trà xanh hữu cơ đóng túi"],
    "090411": ["Hạt tiêu chưa xay, chưa nghiền", "Tiêu đen nguyên hạt xuất khẩu", "Tiêu trắng nguyên hạt loại premium"],
    # Cereals
    "100199": ["Lúa mì loại khác", "Lúa mì nhập khẩu cho công nghiệp bánh mì"],
    "100630": ["Gạo trắng xay xát hoàn toàn, hạt dài", "Gạo Jasmine Việt Nam, đã xát trắng", "Gạo tấm, xay xát một phần, xuất khẩu", "Gạo ST25 Việt Nam, xay xát hoàn toàn", "Gạo trắng hạt dài, xuất khẩu bao 25kg"],
    "120190": ["Đậu nành loại khác", "Hạt đậu tương nhập khẩu cho công nghiệp"],
    # Sugar/Food/Beverages
    "170199": ["Đường trắng tinh luyện từ mía, ICUMSA 45", "Đường thô từ mía, dạng tinh thể", "Đường trắng công nghiệp, bao 50kg", "Đường cát trắng tinh luyện xuất khẩu"],
    "180100": ["Hạt ca cao thô, nguyên hạt", "Ca cao nguyên liệu xuất khẩu"],
    "190531": ["Bánh quy ngọt đóng hộp", "Bánh biscuit ngọt nhập khẩu"],
    "200990": ["Nước ép trái cây hỗn hợp", "Nước ép hoa quả tổng hợp đóng hộp"],
    "220210": ["Nước uống có đường hoặc hương liệu", "Nước ngọt có ga đóng chai"],
    "220300": ["Bia sản xuất từ malt, đóng chai 330ml", "Bia lager lon 500ml", "Bia craft IPA nhập khẩu"],
    "220421": ["Rượu vang đóng chai dưới 2 lít", "Rượu vang đỏ Pháp đóng chai"],
    "220830": ["Rượu whisky nhập khẩu", "Whisky Scotch đóng chai"],
    "240120": ["Thuốc lá đã tách cuống", "Lá thuốc lá chế biến cho công nghiệp"],
    # Minerals/Fuel
    "252329": ["Xi măng Portland loại khác", "Xi măng xây dựng, bao 50kg"],
    "270900": ["Dầu thô, dầu mỏ thô", "Dầu mỏ thô từ giếng khoan"],
    "271012": ["Xăng nhẹ cho ô tô", "Dầu mỏ nhẹ, loại xăng"],
    "271019": ["Dầu diesel cho động cơ", "Dầu mỏ trung bình cho công nghiệp"],
    "271111": ["Khí thiên nhiên hóa lỏng LNG", "LNG cho nhà máy điện"],
    "271600": ["Điện năng xuất khẩu", "Năng lượng điện từ nhà máy"],
    # Chemicals/Pharma
    "280461": ["Silicon tinh khiết 99.99%", "Silicon cấp điện tử"],
    "290531": ["Ethylene glycol công nghiệp", "Ethylene glycol cho sản xuất"],
    "300490": ["Thuốc đóng gói bán lẻ", "Thuốc giảm đau paracetamol 500mg", "Thuốc viên hạ sốt đóng vỉ", "Thuốc kháng sinh đóng chai"],
    "300220": ["Vắc-xin cho người", "Vắc-xin phòng cúm nhập khẩu"],
    "310520": ["Phân bón NPK cho nông nghiệp", "Phân hỗn hợp NPK đóng bao"],
    "330499": ["Mỹ phẩm loại khác", "Sản phẩm làm đẹp đóng gói bán lẻ"],
    "340111": ["Xà phòng cục tắm", "Xà phòng thơm đóng gói bán lẻ"],
    "380891": ["Thuốc trừ sâu nông nghiệp", "Thuốc diệt côn trùng đóng chai"],
    # Plastics/Rubber
    "390110": ["Hạt nhựa polyethylene mật độ thấp LDPE", "LDPE hạt nhựa cho sản xuất màng"],
    "390120": ["Hạt nhựa HDPE mật độ cao", "HDPE hạt nhựa cho ép phun", "Polyethylene mật độ cao cấp ống"],
    "390760": ["Hạt nhựa PET cấp chai", "PET resin cho sản xuất chai nước"],
    "392010": ["Tấm nhựa polyethylene", "Màng PE cho đóng gói"],
    "392321": ["Túi nhựa PE đựng hàng", "Túi rác HDPE 50 lít", "Túi xách nhựa siêu thị"],
    "401110": ["Lốp xe ô tô mới, cỡ 205/55R16", "Lốp xe con mới, 4 mùa", "Lốp radial mới cho xe du lịch", "Lốp xe SUV mới 265/70R16"],
    "401120": ["Lốp xe tải mới, radial", "Lốp xe buýt mới hạng nặng"],
    # Wood/Paper
    "440710": ["Gỗ thông xẻ, dùng cho xây dựng", "Gỗ thông xử lý, kích thước tiêu chuẩn"],
    "470321": ["Bột gỗ hóa học tẩy trắng", "Bột giấy tẩy trắng cho sản xuất giấy"],
    "480256": ["Giấy không tráng phủ, 40-150 g/m²", "Giấy in A4 80gsm"],
    "481910": ["Thùng carton sóng đóng hàng", "Hộp carton sóng 5 lớp"],
    # Textiles/Garments
    "520100": ["Bông thô chưa chải", "Bông ép kiện cho kéo sợi"],
    "520812": ["Vải cotton dệt thoi chưa tẩy", "Vải mộc cotton dệt trơn"],
    "540233": ["Sợi polyester texture dệt vải", "Sợi tổng hợp polyester"],
    "610910": ["Áo thun cotton dệt kim, cổ tròn, nam", "Áo phông nữ cotton, in họa tiết, dệt kim", "Áo thun trẻ em cotton 100% dệt kim", "Áo polo cotton dệt kim tay ngắn"],
    "611030": ["Áo len sợi tổng hợp dệt kim", "Áo khoác len nhân tạo"],
    "620342": ["Quần nam vải cotton dệt thoi", "Quần jeans nam cotton xanh", "Quần chino nam cotton"],
    "620462": ["Quần nữ vải cotton dệt thoi", "Quần nữ cotton thoải mái"],
    "630260": ["Khăn tắm, khăn bếp cotton", "Khăn lau bát cotton cho nhà bếp"],
    "640399": ["Giày đế cao su, mũ tổng hợp", "Giày thể thao đế cao su lưới", "Giày chạy bộ đế cao su"],
    "640510": ["Giày da thật nam", "Giày da cao cấp đi làm"],
    # Metals
    "720839": ["Thép cuộn cán nóng, rộng trên 600mm", "Thép cuộn HRC mác Q235B", "Thép tấm cán nóng dày 3mm"],
    "720917": ["Thép cuộn cán nguội CRC", "Thép tấm cán nguội cho công nghiệp"],
    "730890": ["Kết cấu thép loại khác", "Thép kết cấu xây dựng"],
    "740311": ["Đồng tinh luyện dạng cathode", "Đồng cathode cấp điện tử"],
    "760110": ["Nhôm thỏi chưa pha, 99.7%", "Nhôm billet chưa hợp kim"],
    "760120": ["Nhôm hợp kim dạng thỏi", "Nhôm alloy cho đúc"],
    # Machinery
    "840734": ["Động cơ đánh lửa trên 1000cc", "Động cơ xăng cho ô tô"],
    "841191": ["Phụ tùng động cơ turbo phản lực", "Linh kiện động cơ máy bay"],
    "841810": ["Tủ lạnh kết hợp ngăn đông", "Tủ lạnh hai cửa gia dụng"],
    "841821": ["Tủ lạnh gia dụng máy nén", "Tủ lạnh nhà 300 lít nén khí"],
    "841911": ["Máy nước nóng gas tức thời", "Bình nóng lạnh gas gia dụng"],
    "842139": ["Máy lọc hoặc tinh chế khác", "Máy lọc không khí công nghiệp"],
    "842199": ["Phụ tùng máy lọc khác", "Lõi lọc công nghiệp"],
    "843149": ["Phụ tùng cần cẩu khác", "Linh kiện cẩu và máy nâng"],
    # Electronics
    "847130": ["Máy tính xách tay, màn hình 14 inch", "Laptop di động, nhẹ dưới 10kg", "Laptop cho sinh viên 15.6 inch", "Laptop gaming di động 17 inch", "Laptop 2-in-1 màn hình cảm ứng"],
    "847141": ["Bộ xử lý dữ liệu số khác", "Máy chủ cho trung tâm dữ liệu"],
    "847150": ["Bộ xử lý máy tính", "CPU cho máy tính để bàn"],
    "847170": ["Bộ nhớ máy tính", "Ổ cứng SSD 1TB", "Thiết bị lưu trữ NVMe"],
    "847330": ["Linh kiện máy tính", "Phụ kiện máy tính, RAM, bàn phím"],
    "850440": ["Bộ chuyển đổi điện tĩnh", "Inverter cho hệ thống năng lượng mặt trời"],
    "850710": ["Ắc quy chì-axit cho ô tô", "Ắc quy xe ô tô 12V 60Ah"],
    "850760": ["Pin lithium-ion cho xe điện", "Pin lithium-ion sạc được", "Pin lithium polymer cho điện thoại", "Pin lithium 18650 công nghiệp"],
    "851712": ["Điện thoại thông minh Samsung Galaxy 5G", "Smartphone iPhone 15, 256GB, mở khóa", "Điện thoại thông minh Android 6.7 inch", "Smartphone giá rẻ 4G 64GB", "Điện thoại 5G màn OLED"],
    "852580": ["Camera truyền hình và camera số", "Camera quay phim 4K chuyên nghiệp"],
    "852872": ["TV màu LED 55 inch", "Smart TV 65 inch 4K UHD", "TV OLED màu 77 inch", "TV LCD màu 43 inch HD"],
    "853400": ["Mạch in PCB", "Bo mạch cho thiết bị viễn thông"],
    "854140": ["Linh kiện bán dẫn quang điện, pin mặt trời", "Tấm pin năng lượng mặt trời"],
    "854231": ["Vi xử lý IC 7nm", "Chip CPU cho máy chủ", "Chip GPU xử lý đồ họa"],
    "854232": ["IC bộ nhớ", "Chip nhớ DRAM 16GB"],
    "854239": ["IC điện tử khác", "Chip IC cho hệ thống điều khiển"],
    # Vehicles
    "870120": ["Đầu kéo cho sơ mi rơ moóc", "Xe đầu kéo vận tải"],
    "870323": ["Xe ô tô chở người động cơ xăng 1500-3000cc", "Toyota Camry xăng 2.5L", "Honda Civic đánh lửa 1800cc", "Xe sedan xăng 2000cc mới", "SUV động cơ xăng 2400cc"],
    "870324": ["Xe ô tô chở người động cơ xăng trên 3000cc", "Xe thể thao turbo trên 3 lít"],
    "870332": ["Xe ô tô diesel 1500-2500cc", "Xe bán tải diesel 2.4 lít"],
    "870340": ["Xe ô tô hybrid xăng-điện", "Xe hybrid sạc ngoài"],
    "870380": ["Xe ô tô điện chạy pin, không phát thải", "Xe điện Tesla Model 3, chạy hoàn toàn bằng điện", "Xe ô tô điện BYD pin lithium", "Xe điện nhỏ gọn cho đô thị", "Xe điện SUV tầm xa 400km"],
    "870899": ["Phụ tùng ô tô khác", "Linh kiện xe ô tô nhập khẩu"],
    "871120": ["Xe mô tô 50-250cc", "Xe máy 150cc"],
    # Aircraft/Ships
    "880230": ["Máy bay trọng lượng rỗng 2000-15000kg", "Máy bay chở khách cỡ vừa"],
    "890190": ["Tàu chở hàng khác", "Tàu vận tải hàng hóa quốc tế"],
    # Instruments/Medical
    "900190": ["Sợi quang học viễn thông", "Cáp quang bó sợi"],
    "901839": ["Kim tiêm, ống thông khác", "Thiết bị y tế ống thông mạch máu"],
    "901890": ["Dụng cụ y tế khác", "Thiết bị y tế cho bệnh viện"],
    "902620": ["Dụng cụ đo áp suất", "Đồng hồ đo áp suất công nghiệp"],
    # Furniture/Toys
    "940161": ["Ghế bọc nệm khung gỗ", "Ghế tiếp khách bọc vải khung gỗ teak"],
    "940180": ["Ghế loại khác", "Ghế nhựa văn phòng"],
    "940350": ["Nội thất phòng ngủ gỗ", "Tủ quần áo gỗ sồi phòng ngủ", "Giường gỗ teak queen size"],
    "940360": ["Nội thất gỗ khác", "Bàn làm việc gỗ văn phòng"],
    "950300": ["Xe đạp ba bánh trẻ em", "Xe scooter đồ chơi", "Xe điện đồ chơi cho bé"],
    "950490": ["Trò chơi và đồ chơi khác", "Board game cho gia đình"],
    # Oils/Plants
    "150710": ["Dầu đậu nành thô", "Dầu đậu nành thô cấp công nghiệp"],
    "151190": ["Dầu cọ loại khác", "Dầu cọ tinh luyện nấu ăn"],
    "130219": ["Nhựa thực vật, chiết xuất thực vật", "Chiết xuất thực vật cho công nghiệp"],
    "060110": ["Củ giống cây trồng", "Củ tulip giống trồng"],
    "210111": ["Cà phê hòa tan, chiết xuất cô đặc", "Cà phê bột hòa tan đóng gói"],
}

CHINESE_TEMPLATES = {
    # Meat
    "020130": ["鲜牛肉去骨 冷藏包装 优质级", "新鲜去骨牛肉 餐饮供应", "冷却去骨黄牛肉 真空包装", "鲜牛腱肉 去骨 冷藏运输", "优质鲜牛肉 无骨 出口级别"],
    "020230": ["冷冻去骨牛肉 出口装", "冷冻牛肉去骨 20公斤箱装", "速冻去骨牛肉 加工用", "冷冻无骨牛肉 清真认证"],
    # Fish
    "030389": ["其他冷冻鱼 出口级", "冷冻海鱼 整条装", "冷冻淡水鱼 箱装"],
    "030617": ["冷冻虾仁 去头去壳", "冷冻南美白虾 31/40规格", "冻黑虎虾 去壳去肠线", "冷冻虾 IQF 出口装", "冷冻基围虾 去头去壳 净重"],
    # Dairy
    "040120": ["牛奶 脂肪含量1-6%", "鲜牛奶 低脂 巴氏杀菌", "UHT牛奶 脂肪3.2%"],
    "040210": ["奶粉 脂肪含量不超过1.5%", "脱脂奶粉 工业用", "低脂奶粉 25公斤袋装"],
    "040690": ["其他奶酪", "进口切达奶酪", "马苏里拉奶酪 比萨用"],
    # Vegetables
    "070200": ["新鲜番茄 温室种植", "新鲜樱桃番茄 250克盒装", "优质新鲜番茄 出口级", "有机新鲜番茄 水培种植"],
    "070310": ["新鲜洋葱 冷藏", "新鲜红葱头 出口级", "新鲜大洋葱 优质"],
    "070820": ["新鲜豆类 冷藏", "新鲜四季豆 出口用", "新鲜豌豆 优质"],
    # Fruits
    "080810": ["新鲜苹果 进口 优质级", "新鲜富士苹果 18公斤箱装", "新鲜嘎啦苹果 新西兰产", "新鲜青苹果 澳洲进口"],
    "080300": ["新鲜香蕉 出口级", "新鲜香蕉 园产直销", "新鲜小香蕉 优级"],
    "080510": ["新鲜橙子 优质级", "新鲜脐橙 加州进口", "新鲜蜜橘 箱装"],
    # Coffee/Tea/Spices
    "090111": ["生咖啡豆 阿拉比卡 未烘焙", "绿色咖啡豆 罗布斯塔 未经烘焙", "未烘焙咖啡原料 未脱咖啡因", "埃塞俄比亚咖啡生豆 精品级", "越南罗布斯塔生豆 60公斤麻袋装"],
    "090210": ["绿茶 包装不超过3公斤", "日本绿茶 散叶", "有机绿茶 袋装"],
    "090411": ["胡椒 未研磨", "黑胡椒粒 出口级", "白胡椒粒 优质"],
    # Cereals
    "100199": ["其他小麦", "进口小麦 面包工业用"],
    "100630": ["精米 泰国茉莉香米 5%碎米", "全白米 长粒型 散装出口", "日本寿司米 短粒精米", "越南ST25大米 全精白", "长粒白米 出口装 25公斤袋"],
    "120190": ["其他大豆", "进口大豆 工业榨油用"],
    # Sugar/Food/Beverages
    "170199": ["白砂糖 精炼 ICUMSA 45", "甘蔗原糖 未精炼 散装", "食品级白糖 50公斤袋装", "红糖 未漂白 散装", "工业级蔗糖"],
    "180100": ["可可豆 整粒 生的", "可可原料豆 出口级"],
    "190531": ["甜饼干 盒装", "甜酥饼干 进口"],
    "200990": ["混合果汁 盒装", "复合果汁饮料 零售装"],
    "220210": ["加糖或加味水", "碳酸饮料 瓶装"],
    "220300": ["麦芽啤酒 330毫升瓶装", "拉格啤酒 500毫升罐装", "精酿IPA啤酒 进口"],
    "220421": ["葡萄酒 容器不超过2升", "法国红葡萄酒 瓶装"],
    "220830": ["威士忌 进口", "苏格兰威士忌 瓶装"],
    "240120": ["烟草 已去梗", "加工烟叶 工业用"],
    # Minerals/Fuel
    "252329": ["其他硅酸盐水泥", "建筑水泥 50公斤袋装"],
    "270900": ["石油原油", "原油 钻井产出 炼制用"],
    "271012": ["汽油 车用", "轻质石油 燃料级", "无铅汽油 95号"],
    "271019": ["柴油 发动机用", "中质石油产品 工业用"],
    "271111": ["液化天然气 LNG", "LNG 发电厂用", "液化天然气 船运出口"],
    "271600": ["电能 出口", "发电厂电力"],
    # Chemicals/Pharma
    "280461": ["硅 纯度99.99%以上", "电子级硅材料"],
    "290531": ["乙二醇 工业用", "乙二醇 工业级"],
    "300490": ["其他零售包装药品", "对乙酰氨基酚片 500毫克", "退烧药片 泡罩包装", "抗炎药 瓶装"],
    "300220": ["人用疫苗", "流感疫苗 进口", "新冠疫苗 进口"],
    "310520": ["NPK复合肥 农业用", "三元复合肥 袋装"],
    "330499": ["其他化妆品", "美容产品 零售包装"],
    "340111": ["洗浴香皂 块状", "香皂 零售包装", "有机手工皂"],
    "380891": ["杀虫剂 农业用", "灭虫剂 瓶装"],
    # Plastics/Rubber
    "390110": ["低密度聚乙烯树脂 LDPE", "LDPE粒料 吹膜级"],
    "390120": ["高密度聚乙烯树脂颗粒 HDPE", "HDPE管材级原料 黑色配方", "高密度PE粒料 注塑级", "HDPE原生树脂 自然色", "高密度聚乙烯 容器级"],
    "390760": ["PET瓶级树脂", "聚对苯二甲酸乙二醇酯 瓶级", "PET树脂 25公斤袋装"],
    "392010": ["聚乙烯板片", "PE薄膜 包装用"],
    "392321": ["聚乙烯塑料袋", "HDPE垃圾袋 50升", "超市购物塑料袋"],
    "401110": ["新的轿车轮胎 205/55R16", "新四季轿车轮胎 225/45R17", "新充气子午线轮胎 轿车用", "新SUV轮胎 265/70R16", "新冬季雪地轮胎 轿车用"],
    "401120": ["新卡车轮胎 子午线", "新客车轮胎 重载型"],
    # Wood/Paper
    "440710": ["针叶锯材 建筑用", "松木板材 干燥处理 标准尺寸"],
    "470321": ["漂白化学木浆", "漂白纸浆 造纸用"],
    "480256": ["未涂布纸 40-150克/平方米", "A4打印纸 80克"],
    "481910": ["瓦楞纸箱 包装用", "五层瓦楞纸板箱"],
    # Textiles/Garments
    "520100": ["原棉 未梳理", "棉花打包 纺纱用", "有机原棉 未加工"],
    "520812": ["未漂白平纹棉布", "棉坯布 平纹织造 未漂白"],
    "540233": ["变形聚酯纱线 织布用", "涤纶变形丝"],
    "610910": ["棉质针织T恤 男款 圆领", "女式棉质针织T恤 印花款", "儿童纯棉针织T恤衫", "运动棉针织T恤", "棉针织polo衫 短袖"],
    "611030": ["化纤针织套头衫", "合成纤维针织毛衣"],
    "620342": ["男式棉质机织长裤", "男式牛仔裤 棉质 蓝色", "男式休闲棉裤"],
    "620462": ["女式棉质机织长裤", "女式休闲棉裤"],
    "630260": ["棉质浴巾和厨房毛巾", "棉质洗碗布 厨房用"],
    "640399": ["橡胶底鞋 合成面", "橡胶底运动鞋 网面", "橡胶底跑步鞋"],
    "640510": ["皮面皮鞋 男款", "真皮商务皮鞋 高级"],
    # Metals
    "720839": ["热轧钢卷 宽度超过600毫米", "热轧钢板卷 Q235B", "热轧碳钢板 厚度3毫米"],
    "720917": ["冷轧钢卷 CRC", "冷轧钢板 工业用"],
    "730890": ["其他钢铁结构件", "建筑用钢结构"],
    "740311": ["精炼铜阴极板", "电解铜 电子级"],
    "760110": ["未锻轧铝 纯度99.7%", "铝坯锭 未合金化"],
    "760120": ["未锻轧铝合金", "铝合金铸锭"],
    # Machinery
    "840734": ["点火发动机 超过1000cc", "汽油发动机 汽车用"],
    "841191": ["涡轮喷气发动机零件", "飞机发动机配件"],
    "841810": ["冷藏冷冻组合柜", "双门冰箱冷冻柜 家用"],
    "841821": ["家用压缩式冰箱", "家用冰箱 压缩制冷 300升"],
    "841911": ["燃气即热式热水器", "家用燃气热水器"],
    "842139": ["其他过滤或净化设备", "工业空气净化器"],
    "842199": ["过滤器零件 其他", "工业过滤元件"],
    "843149": ["其他起重机零件", "吊车和起重设备配件"],
    # Electronics
    "847130": ["便携式笔记本电脑 14英寸 16GB内存", "轻薄笔记本电脑 商务用途", "学生用手提电脑 15.6英寸", "游戏笔记本电脑 便携 17英寸", "二合一变形笔记本 触屏"],
    "847141": ["其他数字处理单元", "数据中心服务器"],
    "847150": ["计算机处理器单元", "台式电脑CPU"],
    "847170": ["计算机存储单元", "固态硬盘SSD 1TB", "NVMe存储设备"],
    "847330": ["计算机零部件", "电脑配件 内存 键盘"],
    "850440": ["静态变流器", "逆变器 太阳能系统用"],
    "850710": ["铅酸蓄电池 车用", "汽车蓄电池 12V 60Ah"],
    "850760": ["锂离子电池 18650型号", "电动汽车锂离子电池组 400V", "可充电锂聚合物电池 手机用", "锂离子电池 圆柱形 21700", "磷酸铁锂电池 储能用"],
    "851712": ["智能手机 安卓系统 6.7英寸屏", "苹果iPhone 15手机 256GB 解锁版", "三星Galaxy S24 5G智能手机", "经济型智能手机 4G 64GB", "5G智能手机 OLED屏 512GB"],
    "852580": ["电视摄像机和数码相机", "4K专业摄像机"],
    "852872": ["彩色电视机 LED 55英寸", "智能电视 65英寸 4K超高清", "OLED彩色电视 77英寸", "液晶彩色电视 43英寸 高清", "QLED彩电 50英寸 智能电视"],
    "853400": ["印刷电路板 PCB", "通信设备用电路板"],
    "854140": ["光敏半导体器件 太阳能电池", "太阳能光伏板 发电用"],
    "854231": ["微处理器芯片 7纳米制程", "CPU集成电路 服务器级别", "ARM架构处理器IC 移动SoC", "GPU芯片 图形处理器", "AI加速芯片 神经网络处理"],
    "854232": ["存储器集成电路", "DRAM内存芯片 16GB"],
    "854239": ["其他电子集成电路", "控制系统IC芯片"],
    # Vehicles
    "870120": ["半挂车牵引车", "货运牵引车头"],
    "870323": ["汽油轿车 排量2000cc", "丰田凯美瑞 汽油发动机 2.5L", "本田思域 火花点火 1800cc", "新轿车 汽油 1.6升", "SUV 汽油发动机 2400cc"],
    "870324": ["汽油轿车 排量超过3000cc", "跑车 涡轮增压 超过3升"],
    "870332": ["柴油轿车 1500-2500cc", "柴油皮卡 2.4升"],
    "870340": ["混合动力汽车 油电混合", "插电式混合动力车"],
    "870380": ["纯电动汽车 电池驱动", "特斯拉Model 3 纯电动车", "比亚迪电动汽车 长续航电池", "城市小型电动汽车", "电动SUV 续航400公里"],
    "870899": ["其他汽车零部件", "进口汽车配件"],
    "871120": ["摩托车 50-250cc", "踏板摩托车 150cc"],
    # Aircraft/Ships
    "880230": ["飞机 空重2000-15000公斤", "中型客机"],
    "890190": ["其他货运船舶", "国际贸易货船"],
    # Instruments/Medical
    "900190": ["光导纤维 通信用", "光纤束"],
    "901839": ["其他注射针 导管", "医疗血管导管"],
    "901890": ["其他医疗器械", "医院用医疗设备"],
    "902620": ["压力测量仪器", "工业压力表"],
    # Furniture/Toys
    "940161": ["软垫木框座椅", "木框布艺接待椅"],
    "940180": ["其他座椅", "塑料办公椅"],
    "940350": ["木质卧室家具 床和床头柜", "橡木衣柜 卧室用", "柚木床架 大号"],
    "940360": ["其他木质家具", "木质办公桌"],
    "950300": ["儿童三轮车 脚踏式", "儿童玩具滑板车", "儿童电动玩具车"],
    "950490": ["其他游戏和玩具", "家庭桌游"],
    # Oils/Plants
    "150710": ["大豆原油 精炼用", "大豆粗油 工业级"],
    "151190": ["其他棕榈油", "精炼棕榈油 食用"],
    "130219": ["植物液汁和提取物", "植物提取物 工业用"],
    "060110": ["种植用球茎", "郁金香种球"],
    "210111": ["速溶咖啡 浓缩提取物", "咖啡粉速溶 袋装"],
}

# Generic templates for codes without specific multilingual translations
def generate_generic_descriptions(hs_code, info, count=10):
    """Generate generic product descriptions for any HS code."""
    desc = info["desc"]
    chapter = info["chapter"]
    
    templates = [
        f"{desc}, commercial grade, for import",
        f"{desc} - standard quality, bulk shipment",
        f"{desc}, packed for international trade",
        f"Shipment of {desc.lower()}, FOB terms",
        f"{desc}, meeting international standards",
        f"Imported {desc.lower()}, customs declaration",
        f"{desc}, quantity: various, for wholesale",
        f"Export grade {desc.lower()}, certified",
        f"{desc} for industrial/commercial use",
        f"Trade declaration: {desc.lower()}, new",
        f"{desc}, duty-free zone import",
        f"Consignment of {desc.lower()}, inspected",
        f"{desc}, origin: various countries",
        f"Wholesale lot of {desc.lower()}",
        f"{desc}, packaged per trade requirements",
    ]
    return templates[:count]


def augment_records(records, multiplier=3):
    """Expand dataset size by adding deterministic trade-context variants."""
    if multiplier <= 1:
        return records

    trade_context = [
        "for international wholesale distribution",
        "for customs clearance and import declaration",
        "for regional retail supply chain",
        "for industrial procurement contract",
        "for export under standard commercial terms",
        "for bonded warehouse delivery",
        "for cross-border shipment",
        "for bulk procurement program",
    ]
    shipment_context = [
        "packed in cartons",
        "palletized for container shipment",
        "shipping term CIF",
        "shipping term FOB",
        "with standard commercial invoice and packing list",
        "for 20ft container loading",
        "for mixed-lot cargo",
        "for scheduled maritime transport",
    ]

    expanded = []
    seen = set()

    for row in records:
        base_text = row["text"].strip()
        key = (row["hs_code"], row["language"], base_text)
        if key not in seen:
            expanded.append(row)
            seen.add(key)

        for i in range(multiplier - 1):
            rnd = random.Random(f"{row['hs_code']}|{row['language']}|{base_text}|{i}")
            v1 = trade_context[rnd.randrange(len(trade_context))]
            v2 = shipment_context[rnd.randrange(len(shipment_context))]
            variant = f"{base_text}, {v1}, {v2}."
            variant_key = (row["hs_code"], row["language"], variant)
            if variant_key in seen:
                continue
            new_row = dict(row)
            new_row["text"] = variant
            expanded.append(new_row)
            seen.add(variant_key)

    return expanded


def load_official_hs_subheadings():
    """Load 6-digit official HS subheadings from datasets/harmonized-system."""
    hs_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "harmonized-system",
        "harmonized-system.csv",
    )
    rows = []
    if not os.path.exists(hs_path):
        return rows

    with open(hs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level") != "6":
                continue
            hscode = row.get("hscode", "").strip()
            desc = row.get("description", "").strip()
            if len(hscode) != 6 or not hscode.isdigit() or not desc:
                continue
            rows.append((hscode, desc))
    return rows


def load_official_chapter_labels():
    """Build chapter code -> human label map from official HS level-2 rows."""
    hs_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "harmonized-system",
        "harmonized-system.csv",
    )
    labels = {}
    if not os.path.exists(hs_path):
        return labels

    with open(hs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level") != "2":
                continue
            code = row.get("hscode", "").strip()
            desc = row.get("description", "").strip()
            if len(code) == 2 and code.isdigit() and desc:
                labels[code] = desc
    return labels


OFFICIAL_CHAPTER_LABELS = load_official_chapter_labels()


def make_record(text, hs_code, chapter_name, hs_desc, language):
    """Create a normalized training row with human chapter label + chapter code."""
    hs_code = str(hs_code).zfill(6)
    chapter_2 = hs_code[:2]
    chapter_code = f"HS {chapter_2}"
    human_label = OFFICIAL_CHAPTER_LABELS.get(chapter_2, chapter_name)
    return {
        "text": text,
        "hs_code": hs_code,
        "hs_chapter": human_label,
        "hs_chapter_code": chapter_code,
        "hs_chapter_name": chapter_name,
        "hs_desc": hs_desc,
        "language": language,
    }


def _realistic_product_text(desc, hs_code, rng):
    """Generate a realistic product description based on chapter context.

    Instead of wrapping the raw HS description with 'imported goods:' etc,
    this produces text that looks like what appears on real commercial
    invoices, packing lists, and trade declarations — with brand names,
    model numbers, specs, quantities, and packaging details.
    """
    chapter = int(hs_code[:2])
    dl = desc.lower()

    # ── Chapter-specific generators ──────────────────────────────────
    # Each returns a list of realistic text variants.  We pick one at
    # random for each call.

    # Ch 01-05  Live animals / Animal products
    if chapter <= 5:
        grades = ["Grade A", "Premium", "Select", "Choice", "HALAL certified",
                  "USDA inspected", "EU-approved", "organic"]
        packs = ["vacuum sealed", "in 20kg cartons", "bulk IQF", "10kg master carton",
                 "shrink-wrapped", "MAP packaged", "blast frozen", "cryovac packed"]
        opts = [
            f"{rng.choice(grades)} {dl}, {rng.choice(packs)}",
            f"{dl}, sourced from certified farms, {rng.choice(packs)}",
            f"Fresh {dl} for wholesale, {rng.choice(grades).lower()}",
            f"{dl}, {rng.choice(packs)}, export quality",
            f"Commercial lot: {dl}, {rng.choice(grades).lower()}, new season",
        ]
        return rng.choice(opts)

    # Ch 06-14  Vegetable products
    if chapter <= 14:
        origins = ["Thailand", "Vietnam", "Brazil", "Colombia", "India",
                   "Kenya", "Ethiopia", "Indonesia", "Mexico", "Peru"]
        packs = ["in 25kg PP bags", "bulk in 1-tonne FIBC", "50kg jute bags",
                 "5kg retail packs", "cardboard trays", "netted bags"]
        opts = [
            f"{dl}, origin {rng.choice(origins)}, {rng.choice(packs)}",
            f"Organic {dl} from {rng.choice(origins)}, new harvest",
            f"{dl}, {rng.choice(packs)}, grade I, for supermarket",
            f"Premium {dl}, single origin {rng.choice(origins)}",
            f"{dl}, freshly harvested, {rng.choice(packs)}, export grade",
        ]
        return rng.choice(opts)

    # Ch 15-24  Prepared foodstuffs / beverages
    if chapter <= 24:
        brands = ["Nestlé", "Unilever", "Kraft Heinz", "Mondelez", "Mars",
                  "PepsiCo", "Coca-Cola", "AB InBev", "Diageo", "JBS"]
        packs = ["6-pack carton", "24x330ml case", "12x1L box", "500g retail tin",
                 "bulk 200L drum", "20kg bag", "5L jerrycan", "pallet of 48 cases"]
        opts = [
            f"{rng.choice(brands)} brand {dl}, {rng.choice(packs)}",
            f"{dl}, retail-ready packaging, {rng.choice(packs)}",
            f"Commercial supply of {dl}, {rng.choice(packs)}",
            f"{dl}, food-grade, {rng.choice(packs)}, shelf-stable",
            f"Private-label {dl}, {rng.choice(packs)}, for retail chain",
        ]
        return rng.choice(opts)

    # Ch 25-27  Mineral products / fuels
    if chapter <= 27:
        specs = ["API gravity 32°", "sulfur content <0.5%", "octane rating 95",
                 "calorific value 45MJ/kg", "purity 99.5%", "mesh size 200"]
        volumes = ["50,000 MT bulk shipment", "20ft ISO tank", "200L steel drums",
                   "tanker vessel cargo", "25kg PP bags, palletized", "1000L IBC"]
        opts = [
            f"{dl}, {rng.choice(specs)}, {rng.choice(volumes)}",
            f"Commodity: {dl}, {rng.choice(volumes)}, industrial grade",
            f"{dl}, {rng.choice(specs)}, for refinery processing",
            f"Bulk {dl}, {rng.choice(volumes)}, origin Middle East",
            f"{dl}, technical grade, {rng.choice(volumes)}",
        ]
        return rng.choice(opts)

    # Ch 28-38  Chemicals / pharmaceuticals
    if chapter <= 38:
        pharma = ["Pfizer", "Roche", "Novartis", "AstraZeneca", "Merck",
                  "GSK", "Johnson & Johnson", "Sanofi", "Bayer", "Abbott"]
        chem = ["BASF", "Dow Chemical", "DuPont", "SABIC", "LyondellBasell",
                "Mitsubishi Chemical", "Shin-Etsu", "Solvay", "Evonik"]
        specs = ["purity 99.9%", "pharma grade", "technical grade", "ACS reagent grade",
                 "USP/NF compliant", "food grade", "industrial grade", "analytical grade"]
        packs = ["25kg fiber drum", "200L HDPE drum", "1000L IBC", "1kg laboratory bottle",
                 "bulk tanker", "5L amber glass bottle", "50kg steel drum"]
        brands = pharma if chapter == 30 or "medic" in dl or "vaccin" in dl else chem
        opts = [
            f"{rng.choice(brands)} {dl}, {rng.choice(specs)}, {rng.choice(packs)}",
            f"{dl}, {rng.choice(specs)}, CAS registered, {rng.choice(packs)}",
            f"Supply of {dl}, {rng.choice(specs)}, {rng.choice(packs)}",
            f"{dl}, {rng.choice(packs)}, SDS included, {rng.choice(specs)}",
            f"Lot: {dl}, batch-tested, {rng.choice(specs)}",
        ]
        return rng.choice(opts)

    # Ch 39-40  Plastics / rubber
    if chapter <= 40:
        types = ["injection moulding grade", "extrusion grade", "blow moulding grade",
                 "film grade", "pipe grade", "FDA-approved food contact grade"]
        brands = ["SABIC", "LyondellBasell", "ExxonMobil Chemical", "Dow", "INEOS",
                  "Formosa Plastics", "Sinopec", "LG Chem", "Braskem"]
        packs = ["25kg PE bags on pallet", "1000kg super sack", "octabin", "500kg gaylord box"]
        opts = [
            f"{rng.choice(brands)} {dl}, {rng.choice(types)}, {rng.choice(packs)}",
            f"{dl}, MFI 2.0, density 0.95, {rng.choice(packs)}",
            f"{dl}, {rng.choice(types)}, virgin resin, {rng.choice(packs)}",
            f"Recycled {dl}, post-consumer, {rng.choice(packs)}",
            f"{dl}, {rng.choice(types)}, natural colour, {rng.choice(packs)}",
        ]
        return rng.choice(opts)

    # Ch 41-43  Leather / furskins
    if chapter <= 43:
        types = ["full-grain", "top-grain", "split leather", "patent leather",
                 "suede", "nubuck", "chrome-tanned", "vegetable-tanned"]
        opts = [
            f"{rng.choice(types).title()} {dl}, Italian origin, for luxury goods",
            f"{dl}, {rng.choice(types)}, dyed black, for shoe manufacturing",
            f"{dl}, {rng.choice(types)}, thickness 1.2mm, in bundles",
            f"Premium {dl}, {rng.choice(types)}, for handbag production",
            f"{dl}, {rng.choice(types)}, bovine origin, 50 sq ft hides",
        ]
        return rng.choice(opts)

    # Ch 44-49  Wood / paper
    if chapter <= 49:
        species = ["pine", "oak", "birch", "eucalyptus", "teak", "spruce", "maple"]
        paper_brands = ["UPM", "Stora Enso", "International Paper", "Sappi", "APP"]
        if chapter <= 46:
            opts = [
                f"{dl}, {rng.choice(species)}, kiln-dried, FSC certified",
                f"{rng.choice(species).title()} {dl}, moisture content <12%",
                f"{dl}, sustainably sourced {rng.choice(species)}, bundled",
                f"Commercial lot: {dl}, {rng.choice(species)}, planed",
                f"{dl}, {rng.choice(species)} wood, for furniture manufacturing",
            ]
        else:
            opts = [
                f"{rng.choice(paper_brands)} {dl}, 80gsm, A4, 500 sheets/ream",
                f"{dl}, coated, 120gsm, for offset printing",
                f"{dl}, uncoated, recycled content 30%, pallet-wrapped",
                f"Bulk order: {dl}, 70gsm, in reams, for commercial printing",
                f"{dl}, bleached, 90gsm, FSC certified, shrink-wrapped",
            ]
        return rng.choice(opts)

    # Ch 50-63  Textiles / apparel
    if chapter <= 63:
        fabrics = ["100% cotton", "65/35 polyester-cotton", "100% polyester",
                   "organic cotton", "linen blend", "silk", "merino wool"]
        brands = ["Nike", "Adidas", "Zara", "H&M", "Uniqlo", "Levi's",
                  "Ralph Lauren", "Calvin Klein", "Tommy Hilfiger", "GAP"]
        sizes = ["S/M/L/XL assorted", "one size", "EU 38-44 range", "US 6-12"]
        if chapter >= 61:  # apparel
            opts = [
                f"{rng.choice(brands)} {dl}, {rng.choice(fabrics)}, {rng.choice(sizes)}",
                f"{dl}, {rng.choice(fabrics)}, men's, {rng.choice(sizes)}, new",
                f"Lot of 500 pcs: {dl}, {rng.choice(fabrics)}, assorted colours",
                f"{dl}, women's, {rng.choice(brands)} brand, retail-packaged",
                f"Children's {dl}, {rng.choice(fabrics)}, printed, dozen-packed",
            ]
        else:
            opts = [
                f"{dl}, {rng.choice(fabrics)}, width 150cm, roll of 100m",
                f"{rng.choice(fabrics)} {dl}, dyed, for garment factory",
                f"{dl}, {rng.choice(fabrics)}, 200 thread count, bleached",
                f"Industrial supply: {dl}, {rng.choice(fabrics)}, greige fabric",
                f"{dl}, {rng.choice(fabrics)}, knitted, tubular, 30kg rolls",
            ]
        return rng.choice(opts)

    # Ch 64-67  Footwear / headgear
    if chapter <= 67:
        brands = ["Nike Air Max 270", "Adidas Ultraboost 24", "New Balance 990v6",
                  "Puma RS-X", "Asics Gel-Kayano 31", "Reebok Classic",
                  "Dr. Martens 1460", "Timberland 6-Inch Boot", "Converse Chuck Taylor"]
        opts = [
            f"{rng.choice(brands)}, {dl}, sizes EU 36-46, new in box",
            f"{dl}, leather upper, rubber sole, 12 pairs/carton",
            f"Athletic {dl}, mesh upper, EVA midsole, {rng.choice(brands).split()[0]} brand",
            f"{dl}, men's, size run 7-13 US, 60 pairs per case",
            f"Women's {dl}, {rng.choice(brands).split()[0]}, retail-boxed",
        ]
        return rng.choice(opts)

    # Ch 68-71  Stone / ceramic / glass / precious metals
    if chapter <= 71:
        if chapter == 71:
            brands = ["Tiffany & Co.", "Cartier", "Pandora", "Swarovski", "De Beers"]
            opts = [
                f"{rng.choice(brands)} {dl}, 18K gold, hallmarked",
                f"{dl}, 925 sterling silver, polished, retail-boxed",
                f"GIA-certified {dl}, 1.5 carat, VS1 clarity, D colour",
                f"Lot of {dl}, 14K white gold, assorted designs",
                f"{dl}, platinum setting, conflict-free diamonds",
            ]
        else:
            opts = [
                f"{dl}, tempered, 6mm thick, 2440x1220mm sheets",
                f"Container of {dl}, for construction, palletized",
                f"{dl}, grade A, polished finish, for interior use",
                f"Ceramic {dl}, glazed, 600x600mm, 20 sqm/pallet",
                f"{dl}, borosilicate, laboratory grade, in crates",
            ]
        return rng.choice(opts)

    # Ch 72-83  Base metals
    if chapter <= 83:
        mills = ["ArcelorMittal", "Nippon Steel", "POSCO", "Baosteel", "Tata Steel",
                 "Nucor", "ThyssenKrupp", "JFE Steel", "SSAB", "Hyundai Steel"]
        specs = ["hot-rolled", "cold-rolled", "galvanized", "stainless 304",
                 "stainless 316L", "electrolytic", "annealed", "pickled"]
        dims = ["thickness 2.0mm, width 1250mm", "coil weight 20MT", "6m lengths",
                "diameter 12mm, in bundles", "gauge 18, 4x8 ft sheets"]
        opts = [
            f"{rng.choice(mills)} {dl}, {rng.choice(specs)}, {rng.choice(dims)}",
            f"{dl}, {rng.choice(specs)}, ASTM A36, {rng.choice(dims)}",
            f"Steel {dl}, {rng.choice(specs)}, EN 10025 certified",
            f"{dl}, {rng.choice(specs)}, mill-direct, {rng.choice(dims)}",
            f"Commercial quality {dl}, {rng.choice(specs)}, test certificate included",
        ]
        return rng.choice(opts)

    # Ch 84-85  Machinery / Electrical equipment
    # Use keyword matching to pick contextually-appropriate brands
    if chapter <= 85:
        # Keyword → brand mapping for realistic product descriptions
        # NOTE: Order matters — more specific patterns MUST come first.
        # "headphone" contains "phone" so must be checked before "phone".
        _brand_map_85 = {
            "headphone|earphone|speaker|loudspeaker|sound": [
                "Bose QuietComfort Ultra Headphones", "Sony WH-1000XM6",
                "Apple AirPods Pro 3", "Samsung Galaxy Buds4 Pro",
                "Sennheiser Momentum 5", "JBL Charge 6 Bluetooth Speaker",
                "Sonos Era 300", "Bang & Olufsen Beoplay H100"],
            "microphone": [
                "Shure SM7dB", "Rode NT1 5th Gen", "Audio-Technica AT2020",
                "Blue Yeti X", "Sennheiser MKE 600", "Neumann U87 Ai"],
            "phone|smartphone|cellular|telephone": [
                "Apple iPhone 17 Pro Max 512GB", "Samsung Galaxy S26 Ultra 256GB",
                "Google Pixel 10 Pro 256GB", "OnePlus 13 Pro",
                "Xiaomi 15 Ultra", "Huawei Pura 80 Pro", "Sony Xperia 1 VII",
                "Nothing Phone (3)", "Motorola Edge 60 Ultra", "OPPO Find X8 Pro"],
            "television|monitor|display": [
                "Sony Bravia XR-77A95L 77\" 4K OLED", "LG OLED65C4 65\" TV",
                "Samsung QN90D 75\" Neo QLED", "TCL 98C955 98\" Mini LED",
                "Hisense U8N 65\" 4K", "Dell UltraSharp U2724D 27\" monitor",
                "ASUS ProArt PA32UCG-K 32\" HDR", "BenQ PD3225U 32\""],
            "camera|photograph|video record": [
                "Canon EOS R5 Mark II", "Sony A7R VI Mirrorless",
                "Nikon Z8 Full-Frame", "Fujifilm X-T6", "GoPro Hero 13 Black",
                "DJI Osmo Action 5 Pro", "Panasonic Lumix S5 IIX"],
            "computer|laptop|notebook|data processing|portable.*<10kg": [
                "Apple MacBook Pro M4 Max 16\"", "Dell XPS 16 9640",
                "Lenovo ThinkPad X1 Carbon Gen 13", "HP EliteBook 860 G11",
                "ASUS ROG Zephyrus G16", "Microsoft Surface Laptop 7",
                "Framework Laptop 16", "Razer Blade 16 2025"],
            "server|data processing.*unit|storage unit": [
                "Dell PowerEdge R760", "HP ProLiant DL380 Gen12",
                "Lenovo ThinkSystem SR650 V3", "Supermicro SYS-621C-TN12R",
                "Samsung 990 PRO 4TB NVMe SSD", "WD Ultrastar DC HC580 22TB"],
            "semiconductor|processor|chip|integrated circuit": [
                "Intel Core Ultra 9 285K processor", "AMD Ryzen 9 9950X CPU",
                "NVIDIA GeForce RTX 5090 24GB GPU", "AMD Radeon RX 9070 XT",
                "Qualcomm Snapdragon 8 Elite", "Apple M4 Ultra chip",
                "Samsung Exynos 2500", "MediaTek Dimensity 9400"],
            "battery|accumulator|cell": [
                "CATL LFP battery cell 280Ah", "Samsung SDI NMC pouch cell",
                "BYD Blade Battery module 150Ah", "LG Energy Solution cylindrical cell",
                "Panasonic 4680 lithium-ion cell", "EVE Energy prismatic cell 314Ah"],
            "lamp|lighting|LED": [
                "Philips Hue White & Color A21", "OSRAM Smart+ LED Strip",
                "Cree XHP70.3 HI LED module", "Signify Interact LED panel",
                "GE Current LED troffer", "Flos IC Ceiling Light"],
            "motor|engine|pump|compressor|turbine": [
                "Siemens SIMOTICS GP motor 75kW", "ABB ACS880 variable speed drive",
                "WEG W22 electric motor 55kW", "Nidec brushless DC motor",
                "Grundfos CRE centrifugal pump", "Atlas Copco GA37+ compressor"],
            "washing|dishwash|dryer|refrigerat|freezer|air condition": [
                "Samsung Bespoke AI Washer WF53BB8700AT",
                "LG WashTower WKG101HWA",
                "Dyson V15 Detect cordless vacuum",
                "Bosch Series 8 dishwasher SMS8YCI03E",
                "Daikin split-type AC FTXZ50N 18000BTU",
                "Carrier 42QHB018D8S ducted unit"],
        }
        _brand_map_84 = {
            "tractor|agricultural|harvester|plough": [
                "John Deere 8R 410", "Caterpillar D6 dozer",
                "Kubota M7-172 tractor", "CLAAS Lexion 8900 combine",
                "New Holland T7.315 HD", "Massey Ferguson 8S.305"],
            "excavat|bulldozer|crane|loader|earth.?mov": [
                "Caterpillar 320 excavator", "Komatsu PC210LC-11",
                "Volvo EC220E excavator", "Liebherr LTM 1300-6.3 crane",
                "Hitachi ZX350LC-7 excavator", "JCB 3CX backhoe loader"],
            "lathe|milling|drilling|CNC|machining|turning": [
                "Haas VF-2SS CNC vertical mill", "DMG Mori NLX 2500/700",
                "Fanuc Robodrill α-D21MiB5", "Mazak QT-250MSY CNC lathe",
                "Okuma GENOS M560-V", "Trumpf TruLaser 3060"],
            "centrifug|filter|distill|reactor|heat exchang": [
                "Alfa Laval ALDEC G3 decanter", "GEA Westfalia separator",
                "Andritz Gouda drum dryer", "Pall Corporation filter housing"],
            "print|copy": [
                "HP DesignJet Z9+ 44\" printer", "Epson SureColor SC-P5370",
                "Canon imagePRESS V1350", "Xerox Versant 4100"],
            "robot|automat": [
                "Fanuc CRX-25iA cobot", "ABB IRB 6700 industrial robot",
                "KUKA KR QUANTEC", "Universal Robots UR20"],
        }
        import re as _re
        brand_map = _brand_map_84 if chapter == 84 else _brand_map_85
        # Find matching brands by keyword
        matched = []
        for pattern, blist in brand_map.items():
            if _re.search(pattern, dl, _re.IGNORECASE):
                matched = blist
                break
        if not matched:
            # Generic fallback for the chapter
            if chapter == 84:
                matched = ["Siemens", "Bosch", "ABB", "Mitsubishi", "Fanuc",
                           "Caterpillar", "Komatsu", "Haas", "DMG Mori"]
            else:
                matched = ["Samsung", "LG", "Sony", "Panasonic", "Philips",
                           "Bosch", "Siemens", "Schneider Electric", "ABB"]
        brand = rng.choice(matched)
        opts = [
            f"{brand}, {dl}, new in original packaging",
            f"{dl}, {brand.split()[0]} brand, with warranty card",
            f"1x {brand}, factory sealed, {dl}",
            f"Lot of 100 units: {dl}, {brand.split(',')[0]}",
            f"{brand} — {dl}, for commercial distribution",
        ]
        return rng.choice(opts)

    # Ch 86-89  Vehicles / aircraft / vessels
    if chapter <= 89:
        if chapter == 87:
            brands = ["Toyota Camry 2026", "Honda CR-V 2025", "Tesla Model Y LR",
                      "BMW X5 xDrive40i", "Mercedes-Benz GLC 300", "Hyundai Tucson HEV",
                      "Ford F-150 Lightning", "Volkswagen ID.4 Pro S",
                      "BYD Seal 82kWh", "Kia EV6 GT-Line"]
        elif chapter == 88:
            brands = ["Boeing 787-9 Dreamliner", "Airbus A350-900", "Embraer E195-E2",
                      "DJI Matrice 350 RTK drone", "Cessna Citation CJ4 Gen2"]
        else:
            brands = ["Maersk container vessel", "bulk carrier 82,000 DWT",
                      "Yamaha outboard motor 300HP", "Boston Whaler 280 Outrage"]
        opts = [
            f"{rng.choice(brands)}, {dl}, new, unregistered",
            f"{dl}, {rng.choice(brands)}, for import and sale",
            f"1 unit: {rng.choice(brands)}, {dl}",
            f"{dl}, brand new {rng.choice(brands)}, with documentation",
            f"Used {rng.choice(brands)}, {dl}, mileage 45,000 km",
        ]
        return rng.choice(opts)

    # Ch 90-92  Instruments / clocks / musical instruments
    if chapter <= 92:
        if chapter == 90:
            brands = ["Zeiss", "Olympus", "Nikon", "Leica", "Shimadzu",
                      "Agilent", "Thermo Fisher", "Roche Diagnostics",
                      "Medtronic", "Siemens Healthineers", "GE HealthCare"]
        elif chapter == 91:
            brands = ["Rolex Submariner", "Omega Seamaster", "Seiko Prospex",
                      "Casio G-Shock", "TAG Heuer Carrera", "Citizen Eco-Drive"]
        else:
            brands = ["Yamaha C7X Grand Piano", "Fender Stratocaster American Pro II",
                      "Gibson Les Paul Standard '60s", "Roland TD-50X V-Drums",
                      "Martin D-28 Acoustic Guitar", "Steinway Model B"]
        opts = [
            f"{rng.choice(brands)}, {dl}, new, factory warranty",
            f"{dl}, {rng.choice(brands)}, professionally calibrated",
            f"1x {rng.choice(brands)}, {dl}, in protective case",
            f"{dl}, manufactured by {rng.choice(brands).split()[0]}, current model year",
            f"Commercial order: {dl}, {rng.choice(brands)}, 10 units",
        ]
        return rng.choice(opts)

    # Ch 93  Arms and ammunition
    if chapter == 93:
        opts = [
            f"{dl}, for military contract, with end-user certificate",
            f"Sporting {dl}, for licensed dealer, serial numbered",
            f"{dl}, deactivated, for museum exhibition",
            f"Competition-grade {dl}, .22LR caliber, cased",
            f"{dl}, law enforcement procurement, with documentation",
        ]
        return rng.choice(opts)

    # Ch 94-96  Furniture / toys / misc manufactured articles
    if chapter <= 96:
        if chapter == 94:
            brands = ["IKEA KALLAX", "Herman Miller Aeron", "Steelcase Leap V2",
                      "Ashley Furniture", "La-Z-Boy", "West Elm", "CB2",
                      "Philips Hue", "OSRAM Smart+", "Flos IC Lights"]
        elif chapter == 95:
            brands = ["LEGO Star Wars UCS set 75375", "Nintendo Switch 2",
                      "PlayStation 6 Digital Edition", "Xbox Series X 2TB",
                      "Hasbro Monopoly", "Mattel Barbie", "Bandai Gundam",
                      "Callaway Paradym Ai Smoke Driver", "TaylorMade Qi35 Irons"]
        else:
            brands = ["Parker Duofold", "Montblanc Meisterstück 149",
                      "BIC Cristal", "Pilot G2", "Staedtler Mars"]
        opts = [
            f"{rng.choice(brands)}, {dl}, new, retail packaged",
            f"{dl}, {rng.choice(brands)}, in original box, for retail sale",
            f"Carton of 24: {dl}, {rng.choice(brands).split()[0]} brand",
            f"{dl}, designed by {rng.choice(brands).split()[0]}, current collection",
            f"Wholesale lot: {dl}, {rng.choice(brands)}, assorted",
        ]
        return rng.choice(opts)

    # Ch 97  Works of art, antiques
    if chapter == 97:
        opts = [
            f"{dl}, original work, with certificate of authenticity",
            f"Contemporary {dl}, by emerging artist, gallery-wrapped",
            f"Antique {dl}, circa 1890, professionally appraised at $12,000",
            f"{dl}, limited edition print, numbered 45/200, framed",
            f"{dl}, for private collection, insured value $25,000",
        ]
        return rng.choice(opts)

    # Fallback for any chapter not explicitly covered
    contexts = ["new, commercial grade", "for wholesale distribution",
                "factory sealed, with documentation", "bulk order, 500 units",
                "retail-ready, individually packaged"]
    return f"{dl}, {rng.choice(contexts)}"


def add_official_hs_examples(data):
    """Append realistic product descriptions for every 6-digit HS code.

    Uses chapter-aware generators to produce text that resembles real
    commercial invoices, packing lists, and customs declarations — with
    brand names, model numbers, specs, and packaging details.
    """
    variants = max(5, int(os.getenv("OFFICIAL_HS_VARIANTS", "5")))
    rng = random.Random(42)  # deterministic for reproducibility

    official_rows = load_official_hs_subheadings()
    if not official_rows:
        return data

    # Track which codes already have enough examples
    code_counts = {}
    for row in data:
        code_counts[row["hs_code"]] = code_counts.get(row["hs_code"], 0) + 1

    for hs_code, desc in official_rows:
        info = HS_CODES.get(hs_code, {"desc": desc, "chapter": f"HS {hs_code[:2]}"})
        existing = code_counts.get(hs_code, 0)
        needed = max(0, variants - existing)
        for _ in range(needed):
            text = _realistic_product_text(desc, hs_code, rng)
            data.append(make_record(text, hs_code, info["chapter"], info["desc"], "en"))

    # De-duplicate exact same (text, hs_code, language) rows.
    unique = {}
    for row in data:
        key = (row["text"], row["hs_code"], row["language"])
        unique[key] = row
    return list(unique.values())


def generate_dataset():
    """Generate the complete training dataset."""
    data = []
    
    # Add English templates
    for hs_code, templates in ENGLISH_TEMPLATES.items():
        for text in templates:
            data.append(make_record(text, hs_code, HS_CODES[hs_code]["chapter"], HS_CODES[hs_code]["desc"], "en"))
    
    # Add Thai templates
    for hs_code, templates in THAI_TEMPLATES.items():
        for text in templates:
            data.append(make_record(text, hs_code, HS_CODES[hs_code]["chapter"], HS_CODES[hs_code]["desc"], "th"))
    
    # Add Vietnamese templates
    for hs_code, templates in VIETNAMESE_TEMPLATES.items():
        for text in templates:
            data.append(make_record(text, hs_code, HS_CODES[hs_code]["chapter"], HS_CODES[hs_code]["desc"], "vi"))
    
    # Add Chinese templates
    for hs_code, templates in CHINESE_TEMPLATES.items():
        for text in templates:
            data.append(make_record(text, hs_code, HS_CODES[hs_code]["chapter"], HS_CODES[hs_code]["desc"], "zh"))
    
    # Fill remaining HS codes with generic descriptions
    for hs_code, info in HS_CODES.items():
        if hs_code not in ENGLISH_TEMPLATES:
            generic = generate_generic_descriptions(hs_code, info, 10)
            for text in generic:
                data.append(make_record(text, hs_code, info["chapter"], info["desc"], "en"))

    # Add official 6-digit HS subheadings from datasets/harmonized-system.
    data = add_official_hs_examples(data)
    
    # Expand dataset with synthetic trade-context variants.
    # Keep default moderate for hosted startup time.
    multiplier = int(os.getenv("DATA_AUG_MULTIPLIER", "2"))
    multiplier = max(1, multiplier)
    data = augment_records(data, multiplier=multiplier)

    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    return data


def main():
    data = generate_dataset()
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "training_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "hs_code",
                "hs_chapter",
                "hs_chapter_code",
                "hs_chapter_name",
                "hs_desc",
                "language",
            ],
        )
        writer.writeheader()
        writer.writerows(data)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "training_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Save HS code reference
    ref_path = os.path.join(output_dir, "hs_codes_reference.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(HS_CODES, f, ensure_ascii=False, indent=2)
    
    # Print stats
    print(f"Total examples: {len(data)}")
    unique_codes = set(d["hs_code"] for d in data)
    print(f"Unique HS codes: {len(unique_codes)}")
    
    lang_counts = {}
    for d in data:
        lang_counts[d["language"]] = lang_counts.get(d["language"], 0) + 1
    print(f"Language distribution: {lang_counts}")
    
    chapter_counts = {}
    for d in data:
        chapter_counts[d["hs_chapter"]] = chapter_counts.get(d["hs_chapter"], 0) + 1
    print(f"Chapter distribution: {dict(sorted(chapter_counts.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
