def sentence2vec(sentence, model, tf_idf, num_features=300):
	error_bool = True
	total_word_vec=np.zeros(num_features)
	total_tf_idf_weight=0
	for word in arbdoc_sentences[10]:
		try:
			tf_idf_weight=tf_idf[word]
			word_vec=model[word]
		except:
			error_bool = True
		if not error_bool:
			total_word_vec+=tf_idf_weight*word_vec
			total_tf_idf_weight+=tf_idf_weight
		else:
			error_bool = False
	return(total_word_vec/total_tf_idf_weight)