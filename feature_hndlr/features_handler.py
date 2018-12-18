def extrai_feature(caminho,feature):
	labels = []
	with open(caminho+feature+'_labels.csv', 'r') as labels_csv:  
		for row in csv.reader(labels_csv,quoting=csv.QUOTE_NONNUMERIC):      
			labels.append(row)    

	labels = np.array(labels)
	print('\n################## Extraindo as features do csv '+feature+' ##################')
	print('quant de labels: ',len(labels))
	print('labels.shape: ',labels.shape)

	features = []
	with open(caminho+feature+'_tratado.csv', 'r') as features_csv:
		for row in csv.reader(features_csv,quoting=csv.QUOTE_NONNUMERIC):
			features.append(row)

	#correção do tamanho dos comentários 
	q = len(feature)
	c = int(q/2)*'#'
	f = (int(q/2) + q%2)*'#'

	features = np.array(features)
	print('\nquant de amostras (para '+feature+'): ',len(features))
	print('features.shape: ',features.shape)
	print('#########################'+c+' Fim da extração '+f+'#########################\n')
	return features, labels