# (nazwa po polsku, opis po angielsku dla CLIP)
LABELS = [
    # psy
    ("dalmatyńczyk", "Dalmatian dog"), ("husky syberyjski", "Siberian Husky"),
    ("malamut", "Alaskan Malamute"), ("samojed", "Samoyed dog"),
    ("owczarek niemiecki", "German Shepherd dog"), ("owczarek australijski", "Australian Shepherd dog"),
    ("owczarek szetlandzki", "Shetland Sheepdog"), ("border collie", "Border Collie"),
    ("owczarek podhalański", "Polish Tatra Sheepdog"), ("labrador retriever", "Labrador Retriever"),
    ("golden retriever", "Golden Retriever"), ("nowofundland", "Newfoundland dog"),
    ("berneński pies pasterski", "Bernese Mountain Dog"), ("dog niemiecki", "Great Dane"),
    ("bokser", "Boxer dog"), ("dobermann", "Dobermann"), ("rottweiler", "Rottweiler"),
    ("amstaff", "American Staffordshire Terrier"), ("bulterier", "Bull Terrier"),
    ("buldog francuski", "French Bulldog"), ("buldog angielski", "English Bulldog"), ("mops", "Pug"),
    ("beagle", "Beagle"), ("cocker spaniel", "Cocker Spaniel"),
    ("cavalier king charles spaniel", "Cavalier King Charles Spaniel"), ("jamnik", "Dachshund"),
    ("jack russell terrier", "Jack Russell Terrier"), ("yorkshire terrier", "Yorkshire Terrier"),
    ("west highland white terrier", "West Highland White Terrier"), ("sznaucer", "Schnauzer"),
    ("shih tzu", "Shih Tzu"), ("maltańczyk", "Maltese dog"), ("bichon frise", "Bichon Frise"),
    ("pudel", "Poodle"), ("szpic miniaturowy", "Pomeranian dog"), ("chihuahua", "Chihuahua"),
    ("papillon", "Papillon dog"), ("corgi", "Pembroke Welsh Corgi"), ("shiba inu", "Shiba Inu"),
    ("akita", "Akita dog"), ("weimaraner", "Weimaraner"), ("wyżeł węgierski", "Vizsla"),
    ("chart", "Greyhound"), ("whippet", "Whippet"), ("pies mieszaniec", "mixed-breed dog"),
    # koty
    ("kot brytyjski", "British Shorthair cat"), ("kot perski", "Persian cat"), ("maine coon", "Maine Coon cat"),
    ("ragdoll", "Ragdoll cat"), ("kot syjamski", "Siamese cat"), ("sfinks", "Sphynx cat"),
    ("kot bengalski", "Bengal cat"), ("kot norweski leśny", "Norwegian Forest Cat"),
    ("kot szkocki zwisłouchy", "Scottish Fold cat"), ("kot rosyjski niebieski", "Russian Blue cat"),
    ("kot europejski", "domestic tabby cat"), ("czarny kot", "black cat"), ("rudy kot", "orange tabby cat"),
    ("biały kot", "white cat"),
    # inne
    ("królik", "rabbit"), ("świnka morska", "guinea pig"), ("chomik", "hamster"), ("fretka", "ferret"),
    ("szczur", "pet rat"), ("jeż", "hedgehog"), ("koń", "horse"), ("papuga", "parrot"),
    ("żółw", "turtle"), ("jaszczurka", "lizard"),
]
TEMPLATES = ["a photo of a {}.", "a photo of a {}, a type of pet.", "a close-up photo of a {}.",
             "a photo of a person holding a {}."]
