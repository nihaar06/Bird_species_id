"""
Species lookup table — BirdCLEF folder code -> common / scientific name.

Used throughout the UI so predictions are shown as
"Acadian Flycatcher (acafly)" rather than raw "acafly".
"""

SPECIES_LOOKUP = {
    "acafly":  {"common": "Acadian Flycatcher",        "scientific": "Empidonax virescens"},
    "acowoo":  {"common": "Acorn Woodpecker",           "scientific": "Melanerpes formicivorus"},
    "aldfly":  {"common": "Alder Flycatcher",           "scientific": "Empidonax alnorum"},
    "ameavo":  {"common": "American Avocet",            "scientific": "Recurvirostra americana"},
    "amecro":  {"common": "American Crow",              "scientific": "Corvus brachyrhynchos"},
    "amegfi":  {"common": "American Goldfinch",         "scientific": "Spinus tristis"},
    "barant1": {"common": "Barred Antshrike",           "scientific": "Thamnophilus doliatus"},
    "barswa":  {"common": "Barn Swallow",               "scientific": "Hirundo rustica"},
    "batpig1": {"common": "Bat Falcon",                 "scientific": "Falco rufigularis"},
    "bawswa1": {"common": "Bahama Swallow",             "scientific": "Tachycineta cyaneoviridis"},
    "bawwar":  {"common": "Black-and-white Warbler",    "scientific": "Mniotilta varia"},
    "baywre1": {"common": "Bay Wren",                   "scientific": "Cantorchilus nigricapillus"},
    "bbwduc":  {"common": "Black-bellied Whistling-Duck","scientific": "Dendrocygna autumnalis"},
    "bcnher":  {"common": "Black-crowned Night-Heron",  "scientific": "Nycticorax nycticorax"},
    "carchi":  {"common": "Caribbean Elaenia",          "scientific": "Elaenia martinica"},
    "carwre":  {"common": "Carolina Wren",              "scientific": "Thryothorus ludovicianus"},
    "casvir":  {"common": "Cassin's Vireo",             "scientific": "Vireo cassinii"},
    "categr":  {"common": "Cattle Egret",               "scientific": "Bubulcus ibis"},
    "ccbfin":  {"common": "Cuban Bullfinch",            "scientific": "Melopyrrha nigra"},
    "cedwax":  {"common": "Cedar Waxwing",              "scientific": "Bombycilla cedrorum"},
    "chbant1": {"common": "Chestnut-backed Antbird",    "scientific": "Myrmeciza exsul"},
    "chbchi":  {"common": "Chestnut-bellied Chachalaca","scientific": "Ortalis wagleri"},
    "crfpar":  {"common": "Crimson-fronted Parakeet",   "scientific": "Psittacara finschi"},
    "cubthr":  {"common": "Cuban Thrasher",             "scientific": "Mimus gundlachii"},
    "daejun":  {"common": "Dark-eyed Junco",            "scientific": "Junco hyemalis"},
    "dowwoo":  {"common": "Downy Woodpecker",           "scientific": "Dryobates pubescens"},
    "ducfly":  {"common": "Dusky-capped Flycatcher",    "scientific": "Myiarchus tuberculifer"},
    "dusfly":  {"common": "Dusky Flycatcher",           "scientific": "Empidonax oberholseri"},
    "easblu":  {"common": "Eastern Bluebird",           "scientific": "Sialia sialis"},
    "easkin":  {"common": "Eastern Kingbird",           "scientific": "Tyrannus tyrannus"},
    "easmea":  {"common": "Eastern Meadowlark",         "scientific": "Sturnella magna"},
    "laufal1": {"common": "Laughing Falcon",            "scientific": "Herpetotheres cachinnans"},
    "linwoo1": {"common": "Lineated Woodpecker",        "scientific": "Dryocopus lineatus"},
    "littin1": {"common": "Little Tinamou",             "scientific": "Crypturellus soui"},
    "lobdow":  {"common": "Long-billed Dowitcher",      "scientific": "Limnodromus scolopaceus"},
    "mouela1": {"common": "Mountain Elaenia",           "scientific": "Elaenia frantzii"},
    "mouqua":  {"common": "Mountain Quail",             "scientific": "Oreortyx pictus"},
    "rawwre1": {"common": "Rufous-and-white Wren",      "scientific": "Thryophilus rufalbus"},
    "rcatan1": {"common": "Red-crowned Ant Tanager",    "scientific": "Habia rubica"},
    "rebnut":  {"common": "Red-breasted Nuthatch",      "scientific": "Sitta canadensis"},
    "sposan":  {"common": "Spotted Sandpiper",          "scientific": "Actitis macularius"},
    "spotow":  {"common": "Spotted Towhee",             "scientific": "Pipilo maculatus"},
    "subfly":  {"common": "Sulphur-bellied Flycatcher", "scientific": "Myiodynastes luteiventris"},
    "tropar":  {"common": "Tropical Parula",            "scientific": "Setophaga pitiayumi"},
    "tropew1": {"common": "Tropical Pewee",             "scientific": "Contopus cinereus"},
    "tuftit":  {"common": "Tufted Titmouse",            "scientific": "Baeolophus bicolor"},
    "tunswa":  {"common": "Tundra Swan",                "scientific": "Cygnus columbianus"},
    "veery":   {"common": "Veery",                      "scientific": "Catharus fuscescens"},
    "verdin":  {"common": "Verdin",                     "scientific": "Auriparus flaviceps"},
    "wegspa1": {"common": "White-eared Ground-Sparrow",  "scientific": "Melozone leucotis"},
    "wesant1": {"common": "Western Antbird",            "scientific": "Cercomacroides occidentalis"},
    "wesblu":  {"common": "Western Bluebird",           "scientific": "Sialia mexicana"},
}


def get_species_info(code: str) -> dict:
    """Return {'common': ..., 'scientific': ..., 'code': ...} for a folder code."""
    entry = SPECIES_LOOKUP.get(code, {"common": code, "scientific": "—"})
    return {
        "code": code,
        "common": entry["common"],
        "scientific": entry["scientific"],
    }


def common_name(code: str) -> str:
    return SPECIES_LOOKUP.get(code, {"common": code})["common"]
