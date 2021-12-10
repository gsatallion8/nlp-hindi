import stanfordnlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import indicnlp
from indicnlp.tokenize import indic_tokenize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import interpolate
import pandas as pd
import numpy as np

# stanfordnlp.download('hi')

nlp = stanfordnlp.Pipeline(lang='hi',processors='tokenize,lemma')

stopwords_hi = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी',
				'बड़े','मैं','and','रही','आज','लें','आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 
				'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 
				'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 
				'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 
				'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 
				'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 
				'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 
				'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 
				'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 
				'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 
				'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 
				'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 
				'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने', 
				'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 
				'बनि', 'हि', 'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 
				'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 
				'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग', 'हुअ', 'जेसा', 'नहिं']


punctuations = ['nn','n', '।','/', '`', '+', '"', '?', '▁(', '$', '@', '[', '_', "'", '!', ',', ':', '^', '|', ']',
				 '=', '%', '&', '.', ')', '(', '#', '*', '', ';', '-', '}','|','"']

to_be_removed = stopwords_hi + punctuations

documents = pd.read_csv('hin.txt',sep='\t')

documents = documents['Hindi']

# documents = pd.read_csv('hi-train.csv')

# gt_labels = documents['class']

# label_names = np.unique(gt_labels)

# n_classes = len(label_names)

# documents = documents[' news']

lemmatized_tokens = []

for sentence in documents:
	# tokens = indic_tokenize.trivial_tokenize(sentence)

	lemmatized_tokens.append([word.lemma for word in nlp(sentence).sentences[0].words])
	# print(tokens)

def dummy_fun(doc):
	return doc

vectorizer = TfidfVectorizer(
	analyzer='word',
	tokenizer=dummy_fun,
	preprocessor=dummy_fun,
	token_pattern=None,
	stop_words = to_be_removed)

# vectorizer = TfidfVectorizer(tokenizer = indic_tokenize.trivial_tokenize, stop_words = to_be_removed)
# X = vectorizer.fit_transform(documents)

# print(lemmatized_tokens)

X = vectorizer.fit_transform(lemmatized_tokens)
# print(X)

labels_color_map = {0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#008fd5', 4: '#fc4f30', 5: '#e5ae38', 
					6: '#6d904f', 7: '#8b8b8b', 8: '#810f7c'}

true_k = 2
# true_k = 8
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
model.fit(X)

labels = model.predict(X)

X_dense = X.todense()

X_reduced = PCA(n_components = true_k).fit_transform(X_dense)

# print(X_reduced)

fig, ax = plt.subplots()
for index, instance in enumerate(X_reduced):
	pca_comp_1, pca_comp_2 = X_reduced[index]
	color = labels_color_map[labels[index]]
	ax.scatter(pca_comp_1, pca_comp_2, c=color)

# for i in range(true_k):
# 	points = []
# 	for index, instance in enumerate(X_reduced):
# 		pca_comp_1, pca_comp_2 = X_reduced[index]
# 		if labels[index] == i:
# 			points.append([pca_comp_1, pca_comp_2])

# 	hull = ConvexHull(points)

# 	x_hull = np.append(points[hull.vertices,0],
# 					   points[hull.vertices,0][0])
# 	y_hull = np.append(points[hull.vertices,1],
# 					   points[hull.vertices,1][0])

# 	dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
# 	dist_along = np.concatenate(([0], dist.cumsum()))
# 	spline, u = interpolate.splprep([x_hull, y_hull], 
# 									u=dist_along, s=0)
# 	interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
# 	interp_x, interp_y = interpolate.splev(interp_d, spline)
# 	# plot shape
# 	plt.fill(interp_x, interp_y, '--', c=colors[i], alpha=0.2)

plt.show()

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
	print("Cluster %d:" % i),
	for ind in order_centroids[i, :10]:
		print(terms[ind]),
	print

print("\n")
print("Prediction")




Y = vectorizer.transform(["क्रोम ब्राउज़र खोलने के लिए।"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["मेरी बिल्ली भूखी है।"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["Google अनुवाद ऐप अविश्वसनीय है।"])
prediction = model.predict(Y)
print(prediction)
