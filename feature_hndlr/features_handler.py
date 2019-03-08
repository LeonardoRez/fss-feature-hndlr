import numpy as np
import matplotlib.pyplot as plt
import csv

def extrai_feature(caminho,feature, calcado=False, faixa=False):
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

	features = np.array(features)
	print('\nquant de amostras (para '+feature+'): ',len(features))
	print('features.shape: ',features.shape)
	
	if calcado:
		calcados = []
		with open(caminho+feature+'_calcado.csv', 'r') as calcados_csv:
			for row in csv.reader(calcados_csv, quoting=csv.QUOTE_NONNUMERIC):
				calcados.append(row)
		calcados = np.array(calcados)	
		print('\nquant de índices de calçados: ',len(calcados))
		print('calcados.shape: ',calcados.shape)

	if faixa:
		faixas = []
		with open(caminho+feature+'_faixas.csv', 'r') as faixas_csv:
			for row in csv.reader(faixas_csv, quoting=csv.QUOTE_NONNUMERIC):
				faixas.append(row)
		faixas = np.array(faixas)	
		print('\nquant de índices de calçados: ',len(faixas))
		print('faixas.shape: ',faixas.shape)
	
	#correção do tamanho dos comentários 
	q = len(feature)
	c = int(q/2)*'#'
	f = (int(q/2) + q%2)*'#'
	print('#########################'+c+' Fim da extração '+f+'#########################\n')	
	if calcado and faixa:
		return features, labels, calcados, faixas
	if faixa:
		return features, labels, faixas
	if calcado:
		return features, labels, calcados
	return features, labels


def printa_grafico(treinamento, rede, caminho, nome_da_rede, calcado_prefixo='', id_calcado='', faixa_prefixo='', faixa=''):
	SMALL_SIZE = 19
	MEDIUM_SIZE = 22
	BIGGER_SIZE = 40

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('font', weight='normal')          # 
	plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels labelweight
	plt.rc('axes', labelweight='bold')       # fontsize of the x and y labels 
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

	# grafico da acuracia
	plt.plot(treinamento.history['acc'], 'grey')
	plt.plot(treinamento.history['val_acc'], 'black')
	# plt.axis([0, 10000, 0, 1.1])
	plt.title('Acurácia da feature '+nome_da_rede+'', fontsize=22, weight='bold')
	plt.ylabel('Acurácia (%)')
	plt.xlabel('Geração')
	plt.legend(['Treino', 'Teste'], loc='lower right')
	plt.savefig(caminho+rede+'_acc'+calcado_prefixo+str(id_calcado)+faixa_prefixo+str(faixa)+'.pdf', bbox_inches='tight')
	plt.show()   

	plt.plot(treinamento.history['loss'], 'grey')
	plt.plot(treinamento.history['val_loss'],'black')  
	# plt.axis([0, 10000, 0, 1.1])
	plt.title('Perda da feature '+nome_da_rede+'', fontsize=22, weight='bold')
	plt.ylabel('Perda')
	plt.xlabel('Geração')
	plt.legend(['Treino', 'Teste'], loc='center right')
	plt.savefig(caminho+rede+'_loss'+calcado_prefixo+str(id_calcado)+faixa_prefixo+str(faixa)+'.pdf', bbox_inches='tight')
	plt.show()

#função que separa a porção de teste depois que a base já está embaralhada
# train_set, test_set = split_train_test(housing, 0.2)
def split_train_test(data, labels, indices, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]  
	return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices], indices[train_indices], indices[test_indices]