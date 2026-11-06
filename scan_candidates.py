from gensim.models import Word2Vec
import os

desktop = os.path.expanduser("~/Desktop")
usa = Word2Vec.load(os.path.join(desktop, "usa_word2vec.model"))
china = Word2Vec.load(os.path.join(desktop, "china_word2vec.model"))

usa_seeds = [
    "drug", "addiction", "overdose", "treatment", "rehabilitation",
    "trafficking", "offense", "punishment", "prescription", "opioid",
    "heroin", "cocaine", "marijuana", "abuse", "prevention", "recovery"
]

china_seeds = [
    "毒品", "吸毒", "成瘾", "康复", "戒毒", "治疗", "贩毒",
    "处罚", "犯罪", "违法", "预防", "海洛因", "大麻", "滥用",
    "处方", "可卡因"
]

usa_keywords = [
    "drug", "opioid", "heroin", "cocaine", "addict", "rehab",
    "treatment", "overdose", "traffick", "prescri", "marijuana",
    "fentanyl", "methamphetamine", "detox", "substance", "offend",
    "sanction", "criminal", "cannabis", "narcotic", "abuse",
    "recovery", "prevention", "sentenc", "incarcerat", "prison",
    "arrest", "enforce", "penalty", "opiate", "stimulant"
]

china_keywords = [
    "毒", "瘾", "戒", "康复", "吸食", "贩", "走私", "处罚",
    "犯罪", "违法", "治疗", "药", "预防", "海洛因", "可卡因",
    "大麻", "冰毒", "阿片", "成瘾", "强制", "隔离", "矫治",
    "滥用", "处方", "复吸", "脱毒", "戒断", "依赖", "干预",
    "传染", "疾病", "心理", "社区"
]

def usa_filter(word):
    w = word.lower()
    return any(kw in w for kw in usa_keywords)

def china_filter(word):
    return any(kw in word for kw in china_keywords)

usa_candidates = {}
for seed in usa_seeds:
    if seed in usa.wv:
        for word, score in usa.wv.most_similar(seed, topn=100):
            if usa_filter(word) and word not in usa_candidates:
                usa_candidates[word] = round(score, 4)

china_candidates = {}
for seed in china_seeds:
    if seed in china.wv:
        for word, score in china.wv.most_similar(seed, topn=100):
            if china_filter(word) and word not in china_candidates:
                china_candidates[word] = round(score, 4)

print("USA CANDIDATES")
for word in sorted(usa_candidates, key=lambda x: -usa_candidates[x]):
    freq = usa.wv.get_vecattr(word, "count") if hasattr(usa.wv, "get_vecattr") else "n/a"
    print(f"{word}\t{usa_candidates[word]}\t{freq}")

print("\nCHINA CANDIDATES")
for word in sorted(china_candidates, key=lambda x: -china_candidates[x]):
    freq = china.wv.get_vecattr(word, "count") if hasattr(china.wv, "get_vecattr") else "n/a"
    print(f"{word}\t{china_candidates[word]}\t{freq}")