import numpy as np
import matplotlib.pyplot as plt
import csv

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

def printa_grafico(treinamento, rede):
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
	plt.title('Acurácia da feature '+rede+'', fontsize=22, weight='bold')
	plt.ylabel('Acurácia (%)')
	plt.xlabel('Geração')
	plt.legend(['Treino', 'Teste'], loc='lower right')
	plt.savefig('/content/drive/My Drive/test/graficos/'+rede+'_acc.pdf', bbox_inches='tight')
	plt.show()   

	plt.plot(treinamento.history['loss'], 'grey')
	plt.plot(treinamento.history['val_loss'],'black')  
	# plt.axis([0, 10000, 0, 1.1])
	plt.title('Perda da feature '+rede+'', fontsize=22, weight='bold')
	plt.ylabel('Perda')
	plt.xlabel('Geração')
	plt.legend(['Treino', 'Teste'], loc='center right')
	plt.savefig('/content/drive/My Drive/test/graficos/'+rede+'_loss.pdf', bbox_inches='tight')
	plt.show()