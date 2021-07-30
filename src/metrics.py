"""
the function get a (y_true, y_pred) set of pairs and print the metrics for each problem
can be segmentation or classificaton
"""
from numpy.lib.type_check import _real_if_close_dispatcher
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def classification_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("METRICS RESULTS:\n")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tF1-score: {f1}")

def avalia_segm(imgChk, imgGab):
  """(imagem, imagem, int, int, int, int) -> array
  Recebe a imagem imgChk e um gabarito de segment imgGab. 
    
  A função retorna o par dos valores Precision / Recall
  """
  #Processa os totais de True Positives (TP), pts do Objeto na imagem avaliada que são tb pontos de objeto na Imagem Gabarito
  Boo_TP = (imgChk > 0) & (imgGab > 0)
        
  #Processa os totais de Falso Negativos (FN), fundos na imagem que sao pts de objeto no Gabarito       
  Boo_FN = (imgChk == 0) & (imgGab > 0)

  #Processa os totais de Falso Posiivos (FP), bordas na imagem que sao fundos no Gabarito       
  Boo_FP = (imgChk > 0) & (imgGab == 0)        
        
  TP = np.count_nonzero(Boo_TP)
  FN = np.count_nonzero(Boo_FN)
  FP = np.count_nonzero(Boo_FP)
        
  if (TP > 0):
    Val_Recall = float(TP / (TP + FN))
    Val_Precision = float(TP / (TP + FP))
  else:
    Val_Recall = 0.0
    Val_Precision = 0.0
             
  return Val_Precision, Val_Recall

def segmentation_metrics(y_true, y_pred):

    prec = 0
    recall = 0
    for i, j in zip(y_true, y_pred):
        p,r = avalia_segm(i,j)
        prec +=p
        recall+=r
    prec = prec/len(y_true)
    recall = recall/len(y_true)
    print("METRICS RESULTS:\n")
    print(f"\tprecision: {prec}")
    print(f"\treccall of the segmentation: {recall}")