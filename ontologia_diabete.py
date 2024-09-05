from owlready2 import *
import os

class diabetes_ontology:
    def __init__(self):
        self.ontology = get_ontology(os.path.basename("sintomi_diabete.owl")).load()
        self.dict_symptoms = {}
        self.dict_malattie_correlate = {}

    def get_symptoms_descriptions(self):
        dict_symptoms_onto = {}
        
        target_class1 = self.ontology.aumento_fame
        target_class2 = self.ontology.perdita_peso
        target_class3 = self.ontology.poliuria
        target_class4 = self.ontology.prurito
        target_class5 = self.ontology.sete
        target_class6 = self.ontology.stanchezza
        target_class7 = self.ontology.vista_offuscata
        target_class8 = self.ontology.chetoacidosi_diabetica
        target_class9 = self.ontology.ferite_durature
        target_class10 = self.ontology.infezioni_frequenti
        target_class11 = self.ontology.intorpidimento_corpo
        
        for i in self.ontology.individuals():
            
          if target_class1 in i.is_a or target_class2 in i.is_a or target_class3 in i.is_a or target_class4 in i.is_a or target_class5 in i.is_a or target_class6 in i.is_a or target_class7 in i.is_a or target_class8 in i.is_a or target_class9 in i.is_a or target_class10 in i.is_a or target_class11 in i.is_a:      
             dict_symptoms_onto[str(i)] = i.descrizione_sintomo

        for k in dict_symptoms_onto.keys():

            k1 = k
            k1 = k1.replace("sintomi_diabete.istanza_","")
            self.dict_symptoms[k1] = dict_symptoms_onto[k]


    def print_symptoms(self):

        i = 1
        dict_nums_symptoms = {}
        dict_nums_keys = {}

        for k in self.dict_symptoms.keys():

            print("Sintomo [%d]: Nome: %s"%(i,k))
            dict_nums_symptoms[i] = self.dict_symptoms[k]
            dict_nums_keys[i] = k
            i = i + 1

        return dict_nums_symptoms, dict_nums_keys




    def get_malattie_correlate_descriptions(self):
        dict_malattie_correlate_onto = {}
        
        target_class1 = self.ontology.depressione
        target_class2 = self.ontology.disfunzione_erettile
        target_class3 = self.ontology.gastroparesi
        target_class4 = self.ontology.malattie_cardiovascolari
        target_class5 = self.ontology.malattie_pelle
        target_class6 = self.ontology.nefropatia_diabetica
        target_class7 = self.ontology.neuropatia_diabetica
        target_class8 = self.ontology.piede_diabetico
        target_class9 = self.ontology.retinopatia_diabetica
        target_class10 = self.ontology.glaucoma_cataratta
        target_class11 = self.ontology.alzheimer_demenza
        target_class12 = self.ontology.apnea_sonno
        target_class13 = self.ontology.sindrome_ovaio_policistico
        target_class14 = self.ontology.steatosi_epatica
        target_class15 = self.ontology.arteriopatia_periferica

        for i in self.ontology.individuals():
            
            if target_class1 in i.is_a or target_class2 in i.is_a or target_class3 in i.is_a or target_class4 in i.is_a or target_class5 in i.is_a or target_class6 in i.is_a or target_class7 in i.is_a or target_class8 in i.is_a or target_class9 in i.is_a or target_class10 in i.is_a or target_class11 in i.is_a or target_class12 in i.is_a or target_class13 in i.is_a or target_class14 in i.is_a or target_class15 in i.is_a:     
                dict_malattie_correlate_onto[str(i)] = i.descrizione_malattie_correlate

        for k in dict_malattie_correlate_onto.keys():

            k1 = k
            k1 = k1.replace("sintomi_diabete.istanza_","")
            self.dict_malattie_correlate[k1] = dict_malattie_correlate_onto[k]


    def print_malattie_correlate(self):

        i = 1
        dict_nums_malattie_correlate = {}
        dict_nums_keys_malattie_correlate = {}

        for k in self.dict_malattie_correlate.keys():

            print("Sintomo [%d]: Nome: %s"%(i,k))
            dict_nums_malattie_correlate[i] = self.dict_malattie_correlate[k]
            dict_nums_keys_malattie_correlate[i] = k
            i = i + 1

        return dict_nums_malattie_correlate, dict_nums_keys_malattie_correlate


    def trattamenti(self):
        
        response = str(input())
        
       
                
        if response == ("alzheimer") or response == ("demenza"):    
            target = self.ontology.istanza_trattamento_alzheimer_demenza.descrizione_trattamento     
            print(target)
        
        elif response == ("apnea sonno"):     
            target = self.ontology.istanza_trattamento_apnea_sonno.descrizione_trattamento     
            print(target)
            
        elif response == ("arteriopatia periferica"):          
            target = self.ontology.istanza_trattamento_arteriopatia_periferica.descrizione_trattamento     
            print(target)    
        
        elif response == ("depressione"):           
            target = self.ontology.istanza_trattamento_depressione.descrizione_trattamento     
            print(target)
        
        elif response == ("disfunzione erettile"):          
            target = self.ontology.istanza_trattamento_disfunzione_erettile.descrizione_trattamento     
            print(target)
            
            
        elif response == ("gastroparesi"):          
            target = self.ontology.istanza_trattamento_gastroparesi.descrizione_trattamento     
            print(target)  
            
        elif response == ("glaucoma") or response == ("cataratta") or response == ("glaucoma cataratta"):          
            target = self.ontology.istanza_trattamento_glaucoma_cataratta.descrizione_trattamento     
            print(target)  
            
            
        elif response == ("malattie cardiovascolari"):          
            target = self.ontology.istanza_trattamento_malattie_cardiovascolari.descrizione_trattamento     
            print(target)  
            
        
        elif response == ("malattie pelle"):          
            target = self.ontology.istanza_trattamento_malattie_pelle.descrizione_trattamento     
            print(target)  
            
            
        elif response == ("nefropatia diabetica"):          
            target = self.ontology.istanza_trattamento_nefropatia_diabetica.descrizione_trattamento     
            print(target)  
            
            
            
        elif response == ("neuropatia diabetica"):          
            target = self.ontology.istanza_trattamento_neuropatia_diabetica.descrizione_trattamento     
            print(target)  
            
    
        elif response == ("piede diabetico"):          
            target = self.ontology.istanza_trattamento_piede_diabetico.descrizione_trattamento     
            print(target)  
            
        elif response == ("retinopatia diabetica"):          
             target = self.ontology.istanza_trattamento_retinopatia_diabetica.descrizione_trattamento     
             print(target)  
             
             
        elif response == ("sindrome ovaio policistico"):          
             target = self.ontology.istanza_trattamento_sindrome_ovaio_policistico.descrizione_trattamento     
             print(target)      
        
             
        elif response == ("steatosi epatica"):          
            target = self.ontology.istanza_trattamento_steatosi_epatica.descrizione_trattamento     
            print(target)  
        
        else:
            
            print("Non esiste malattia correlata con questo nome\n")
            










