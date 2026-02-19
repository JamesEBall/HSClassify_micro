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

# Thai translations/descriptions
THAI_TEMPLATES = {
    "020130": [
        "เนื้อวัวสด ไม่มีกระดูก สำหรับร้านอาหาร",
        "เนื้อโคสดแช่เย็น ปลอดกระดูก คุณภาพดี",
        "เนื้อวัวไม่มีกระดูก สด ไม่แช่แข็ง",
        "เนื้อโคชิ้นพิเศษ สด ไม่ติดกระดูก",
    ],
    "030617": [
        "กุ้งแช่แข็ง แกะหัว ติดเปลือก",
        "กุ้งขาวแวนนาไมแช่แข็ง ขนาด 31/40",
        "กุ้งกุลาดำแช่แข็ง ปอกเปลือก",
        "กุ้งสดแช่แข็ง สำหรับร้านอาหาร",
    ],
    "070200": [
        "มะเขือเทศสด จากโรงเรือน",
        "มะเขือเทศเชอร์รี่สด แพ็ค 250 กรัม",
        "มะเขือเทศสดคุณภาพดี สำหรับส่งออก",
    ],
    "090111": [
        "เมล็ดกาแฟดิบ อาราบิก้า ยังไม่คั่ว",
        "กาแฟสารดิบ โรบัสต้า ยังไม่ผ่านการคั่ว",
        "เมล็ดกาแฟเขียว ไม่ได้คั่ว ไม่ได้สกัดคาเฟอีน",
    ],
    "100630": [
        "ข้าวหอมมะลิไทย ขัดสี 5% หัก",
        "ข้าวขาวสารขัดมัน บรรจุถุง 25 กก.",
        "ข้าวเหนียว ขัดสีแล้ว พร้อมส่งออก",
        "ข้าวบาสมาติ ขัดสีทั้งเมล็ด คุณภาพดี",
    ],
    "170199": [
        "น้ำตาลทรายขาวบริสุทธิ์ ICUMSA 45",
        "น้ำตาลทรายดิบ สีน้ำตาล จากอ้อย",
        "น้ำตาลทรายขาว บรรจุถุง 50 กก.",
    ],
    "401110": [
        "ยางรถยนต์ใหม่ ขนาด 205/55R16",
        "ยางรถเก๋งใหม่ สำหรับทุกฤดู",
        "ยางรถยนต์นั่งใหม่ แบบเรเดียล",
    ],
    "610910": [
        "เสื้อยืดผ้าฝ้ายถัก คอกลม ผู้ชาย",
        "เสื้อยืดคอตตอน ผู้หญิง พิมพ์ลาย",
        "เสื้อยืดเด็ก ผ้าฝ้าย 100% ถักนิตติ้ง",
    ],
    "847130": [
        "คอมพิวเตอร์แล็ปท็อป จอ 14 นิ้ว",
        "โน้ตบุ๊คพกพา น้ำหนักเบา สำหรับทำงาน",
        "แล็ปท็อปสำหรับนักเรียน จอ 15.6 นิ้ว",
    ],
    "851712": [
        "สมาร์ทโฟน แอนดรอยด์ จอ 6.7 นิ้ว",
        "โทรศัพท์มือถือ ไอโฟน 15 ความจุ 256GB",
        "มือถือ 5G ซัมซุง กาแล็กซี่ S24",
    ],
    "870323": [
        "รถยนต์นั่งเครื่องเบนซิน 2000 ซีซี",
        "รถเก๋งสปาร์คอิกนิชั่น 1500-3000 ซีซี",
        "รถยนต์โตโยต้า คัมรี่ เครื่อง 2.5 ลิตร",
    ],
    "870380": [
        "รถยนต์ไฟฟ้า พลังงานแบตเตอรี่",
        "รถ EV เทสลา โมเดล 3 ไฟฟ้าล้วน",
        "รถยนต์ไฟฟ้า BYD แบตเตอรี่ลิเธียม",
    ],
}

# Vietnamese translations
VIETNAMESE_TEMPLATES = {
    "020130": [
        "Thịt bò tươi không xương, cấp đông mát",
        "Thịt bò không xương tươi cho nhà hàng",
        "Thịt trâu bò tươi ướp lạnh, không có xương",
    ],
    "030617": [
        "Tôm đông lạnh, bóc vỏ, bỏ đầu",
        "Tôm sú đông lạnh, nguyên con",
        "Tôm thẻ chân trắng đông lạnh xuất khẩu",
    ],
    "090111": [
        "Cà phê nhân xanh, chưa rang, Robusta",
        "Hạt cà phê thô Arabica chưa rang",
        "Cà phê nguyên liệu, chưa qua chế biến",
    ],
    "100630": [
        "Gạo trắng xay xát hoàn toàn, hạt dài",
        "Gạo Jasmine Việt Nam, đã xát trắng",
        "Gạo tấm, xay xát một phần, xuất khẩu",
    ],
    "170199": [
        "Đường trắng tinh luyện từ mía, ICUMSA 45",
        "Đường thô từ mía, dạng tinh thể",
    ],
    "610910": [
        "Áo thun cotton dệt kim, cổ tròn, nam",
        "Áo phông nữ cotton, in họa tiết, dệt kim",
    ],
    "847130": [
        "Máy tính xách tay, màn hình 14 inch",
        "Laptop di động, nhẹ dưới 10kg",
    ],
    "851712": [
        "Điện thoại thông minh Samsung Galaxy 5G",
        "Smartphone iPhone 15, 256GB, mở khóa",
    ],
    "870380": [
        "Xe ô tô điện chạy pin, không phát thải",
        "Xe điện Tesla Model 3, chạy hoàn toàn bằng điện",
    ],
}

# Chinese translations
CHINESE_TEMPLATES = {
    "020130": [
        "鲜牛肉去骨 冷藏包装 优质级",
        "新鲜去骨牛肉 餐饮供应",
        "冷却去骨黄牛肉 真空包装",
    ],
    "030617": [
        "冷冻虾仁 去头去壳",
        "冷冻南美白虾 31/40规格",
        "冻黑虎虾 去壳去肠线",
    ],
    "090111": [
        "生咖啡豆 阿拉比卡 未烘焙",
        "绿色咖啡豆 罗布斯塔 未经烘焙",
        "未烘焙咖啡原料 未脱咖啡因",
    ],
    "100630": [
        "精米 泰国茉莉香米 5%碎米",
        "全白米 长粒型 散装出口",
        "日本寿司米 短粒精米",
    ],
    "170199": [
        "白砂糖 精炼 ICUMSA 45",
        "甘蔗原糖 未精炼 散装",
        "食品级白糖 50公斤袋装",
    ],
    "390120": [
        "高密度聚乙烯树脂颗粒 HDPE",
        "HDPE管材级原料 黑色配方",
        "高密度PE粒料 注塑级",
    ],
    "610910": [
        "棉质针织T恤 男款 圆领",
        "女式棉质针织T恤 印花款",
        "儿童纯棉针织T恤衫",
    ],
    "847130": [
        "便携式笔记本电脑 14英寸 16GB内存",
        "轻薄笔记本电脑 商务用途",
        "学生用手提电脑 15.6英寸",
    ],
    "851712": [
        "智能手机 安卓系统 6.7英寸屏",
        "苹果iPhone 15手机 256GB 解锁版",
        "三星Galaxy S24 5G智能手机",
    ],
    "854231": [
        "微处理器芯片 7纳米制程",
        "CPU集成电路 服务器级别",
        "ARM架构处理器IC 移动SoC",
    ],
    "870323": [
        "汽油轿车 排量2000cc",
        "丰田凯美瑞 汽油发动机 2.5L",
        "本田思域 火花点火 1800cc",
    ],
    "870380": [
        "纯电动汽车 电池驱动",
        "特斯拉Model 3 纯电动车",
        "比亚迪电动汽车 长续航电池",
    ],
    "850760": [
        "锂离子电池 18650型号",
        "电动汽车锂离子电池组 400V",
        "可充电锂聚合物电池 手机用",
    ],
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


def add_official_hs_examples(data):
    """Append official HS descriptions as extra English training examples."""
    variants = max(1, int(os.getenv("OFFICIAL_HS_VARIANTS", "1")))

    base_templates = [
        "{desc}",
        "{desc}, customs declaration entry",
        "imported goods: {desc}",
        "{desc}, commercial invoice description",
    ]

    official_rows = load_official_hs_subheadings()
    if not official_rows:
        return data

    for hs_code, desc in official_rows:
        info = HS_CODES.get(hs_code, {"desc": desc, "chapter": f"HS {hs_code[:2]}"})
        for i in range(variants):
            template = base_templates[min(i, len(base_templates) - 1)]
            text = template.format(desc=desc.lower() if i > 0 else desc)
            data.append({
                "text": text,
                "hs_code": hs_code,
                "hs_chapter": info["chapter"],
                "hs_desc": info["desc"],
                "language": "en",
            })

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
            data.append({
                "text": text,
                "hs_code": hs_code,
                "hs_chapter": HS_CODES[hs_code]["chapter"],
                "hs_desc": HS_CODES[hs_code]["desc"],
                "language": "en"
            })
    
    # Add Thai templates
    for hs_code, templates in THAI_TEMPLATES.items():
        for text in templates:
            data.append({
                "text": text,
                "hs_code": hs_code,
                "hs_chapter": HS_CODES[hs_code]["chapter"],
                "hs_desc": HS_CODES[hs_code]["desc"],
                "language": "th"
            })
    
    # Add Vietnamese templates
    for hs_code, templates in VIETNAMESE_TEMPLATES.items():
        for text in templates:
            data.append({
                "text": text,
                "hs_code": hs_code,
                "hs_chapter": HS_CODES[hs_code]["chapter"],
                "hs_desc": HS_CODES[hs_code]["desc"],
                "language": "vi"
            })
    
    # Add Chinese templates
    for hs_code, templates in CHINESE_TEMPLATES.items():
        for text in templates:
            data.append({
                "text": text,
                "hs_code": hs_code,
                "hs_chapter": HS_CODES[hs_code]["chapter"],
                "hs_desc": HS_CODES[hs_code]["desc"],
                "language": "zh"
            })
    
    # Fill remaining HS codes with generic descriptions
    for hs_code, info in HS_CODES.items():
        if hs_code not in ENGLISH_TEMPLATES:
            generic = generate_generic_descriptions(hs_code, info, 10)
            for text in generic:
                data.append({
                    "text": text,
                    "hs_code": hs_code,
                    "hs_chapter": info["chapter"],
                    "hs_desc": info["desc"],
                    "language": "en"
                })

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
        writer = csv.DictWriter(f, fieldnames=["text", "hs_code", "hs_chapter", "hs_desc", "language"])
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
