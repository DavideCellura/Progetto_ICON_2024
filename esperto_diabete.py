from experta import *
from colorama import Fore
from data_diabete import diabetes_data
from csp_laboratorio import laboratory_csp
from ontologia_diabete import diabetes_ontology


DIABETES_RANDOM_TEST = 11.1
DIABETES_FASTING_TEST = 7


def reset_color():
    print(Fore.RESET)


def valid_response(response: str): 

    valid = False
    response = response.lower()

    if response == "si" or response == "no":
        valid = True

    return valid


def valid_random_test_blood_value(test_value: float):

    valid = False

    if test_value > 3.9:
        valid = True

    return valid


class esperto_diabete(KnowledgeEngine):

    @DefFacts()
    def _initial_action(self):
        yield Fact(inizio="si")
        self.mean_diabetes_tests = diabetes_data().get_medium_values_diabetes()
        self.number_prints = 0
        self.flag_no_symptoms = 0

        self.lab_glucose_analysis = laboratory_csp("Laboratorio Analisi degli zuccheri nel sangue")
        self.lab_glucose_analysis.addConstraint(lambda day,hours: hours >= 8 and hours <= 14 if day == "lunedi" else hours >= 15 and hours <= 20 if day == "giovedi" else None ,["day","hours"])


    def print_facts(self):
        print("\n\nL'agente ragiona con i seguenti fatti: \n")
        print(self.facts)

    def _prototype_lab_booking(self, ask_text: str, lab_selected: laboratory_csp):
        print("Hai avuto la prescrizione per %s, vuoi prenotare presso uno studio convenzionato? [si/no]" %ask_text)
        response = str(input())

        while valid_response(response) == False:
            print("Hai avuto la prescrizione per %s, vuoi prenotare presso uno studio convenzionato? [si/no]"%ask_text)
            response = str(input())
        
        if response == "si":
            first, last = lab_selected.get_availability()

            print("Insersci un turno inserendo il numero del turno associato")
            turn_input = int(input())

            while turn_input < first or turn_input > last:
                print("Insersci un turno inserendo il numero del turno associato")
                turn_input = int(input())
            
            lab_selected.print_single_availability(turn_input)


    def _prototype_ask_symptom(self, ask_text: str, fact_declared: Fact):

        print(ask_text)
        response = str(input())

        while valid_response(response) == False:
            print(ask_text)
            response = str(input())
        if response == "si":
            self.declare(fact_declared)

        return response

    @Rule(Fact(inizio="si"))
    def rule_1(self):
        print(Fore.CYAN + "\nInizio della diagnosi...\n")
        reset_color()
        self.declare(Fact(chiedi_sintomi="si"))

    @Rule(Fact(chiedi_esami_glicemia="si"))
    def rule_2(self):
        print("Hai eseguito un test casuale del sangue?")
        casual_blood_test = str(input())

        while valid_response(casual_blood_test) == False:
            print("Hai eseguito un test casuale del sangue?")
            casual_blood_test = str(input())

        print("Hai eseguito un test del sangue a digiuno?")
        fasting_blood_test = str(input())

        while valid_response(fasting_blood_test) == False:
            print("Hai eseguito un test del sangue a digiuno?")
            fasting_blood_test = str(input())

        if casual_blood_test == "si":
            self.declare(Fact(test_casuale_sangue="si"))
        else:
            self.declare(Fact(test_casuale_sangue="no"))

        if fasting_blood_test == "si":
            self.declare(Fact(test_digiuno_sangue="si"))
        else:
            self.declare(Fact(test_digiuno_sangue="no"))

        if fasting_blood_test == "no" and casual_blood_test == "no":
            self.declare(Fact(prescrizione_esami_sangue="si"))

    @Rule(Fact(test_casuale_sangue="si"))
    def rule_3(self):
        print(
            "Inserisci il valore del test casuale espresso in millimoli su litro [mmol/L]")
        test_value = float(input())

        while valid_random_test_blood_value(test_value) == False:
            print("Inserisci il valore del test casuale espresso in millimoli su litro [mmol/L]")
            test_value = float(input())

        if test_value > DIABETES_RANDOM_TEST:
            self.declare(Fact(glicemia_casuale_alta="si"))

        else:
            self.declare(Fact(glicemia_normale="si"))

    @Rule(Fact(test_digiuno_sangue="si"))
    def rule_4(self):
        print(
            "Inserisci il valore del test a digiuno espresso in millimoli su litro [mmol/L]")
        test_value = float(input())

        while valid_random_test_blood_value(test_value) == False:
            print(
                "Inserisci il valore del test a digiuno espresso in millimoli su litro [mmol/L]")
            test_value = float(input())

        if test_value > DIABETES_FASTING_TEST:
            self.declare(Fact(glicemia_digiuno_alta="si"))
        else:
            self.declare(Fact(glicemia_normale="si"))

    @Rule(Fact(chiedi_sintomi="si"))
    def rule_5(self):

        r1 = self._prototype_ask_symptom("Ti senti molto assetato di solito (sopratutto di notte) ? [si/no]", Fact(sete="si"))
        r2 = self._prototype_ask_symptom("Ti senti molto stanco? [si/no]", Fact(stanchezza="si"))
        r3 = self._prototype_ask_symptom("Stai perdendo peso e massa muscolare? [si/no]", Fact(perdita_peso="si"))
        r4 = self._prototype_ask_symptom("Senti prurito? [si/no]", Fact(prurito="si"))
        r5 = self._prototype_ask_symptom("Hai la vista offuscata? [si/no]", Fact(vista_offuscata="si"))
        r6 = self._prototype_ask_symptom("Consumi spesso bevande/alimenti zuccherati? [si/no]", Fact(bevande_zuccherate="si"))
        r7 = self._prototype_ask_symptom("Hai fame costantemente? [si/no]", Fact(fame_costante="si"))
        r8 = self._prototype_ask_symptom("Hai bisogno costante di urinare? [si/no]", Fact(poliuria="si"))

        if r1 == "no" and r2 == "no" and r3 == "no" and r4 == "no" and r5 == "no" and r6 == "no" and r7 == "no" and r8 == "no":
            self.flag_no_symptoms = 1

        self.declare(Fact(chiedi_imc="si"))


    @Rule(Fact(chiedi_imc="si"))
    def ask_bmi(self):

        medium_bmi_diabetes = self.mean_diabetes_tests['bmi']

        print(Fore.CYAN + "\n\nInserisci l'altezza in centimetri")
        reset_color()
        height = float(input())

        while height < 135 or height > 220:
            print(Fore.CYAN + "Inserisci di nuovo l'altezza in centimetri")
            reset_color()
            height = float(input())

        print(Fore.CYAN + "Inserisci il peso in kilogrammi")
        reset_color()
        weight = float(input())

        while weight < 30 or weight > 250:
            print(Fore.CYAN + "Inserisci DI NUOVO il peso in kilogrammi")
            reset_color()
            weight = float(input())


        height=height/100
        bmi = round(weight/(height*height), 3)

        if bmi >= medium_bmi_diabetes:
            print(Fore.YELLOW + "Il valore del tuo indice di massa corporea pari a %f e' superiore al valore medio di indice di massa corporea dei diabetici.Sei sovrappeso" % bmi)
            reset_color()

   

    @Rule(OR(Fact(fame_costante="si"), Fact(bevande_zuccherate="si")))
    def exam_1(self):
        self.declare(Fact(chiedi_esami_glicemia="si"))


    @Rule(AND(Fact(sete="si"), Fact(stanchezza="si"), Fact(perdita_peso="si"), Fact(prurito="si"), Fact(vista_offuscata="si"), Fact(bevande_zuccherate="si"), Fact(fame_costante="si"), Fact(poliuria="si")))
    def all_diabetes_symptoms(self):
        print("Sembra che tu abbia TUTTI i sintomi del diabete")
        self.declare(Fact(tutti_sintomi="si"))
        self.declare(Fact(chiedi_esami_glicemia="si"))
        

    @Rule(AND(Fact(sete="si"), Fact(stanchezza="si"), Fact(perdita_peso="si"), Fact(prurito="si"), Fact(vista_offuscata="si"), Fact(bevande_zuccherate="si"), Fact(fame_costante="si"), Fact(poliuria="si")), Fact(glicemia_digiuno_alta="si"), Fact(glicemia_casuale_alta="si"))
    def all_diabetes_diagnosis_3(self):
        print(Fore.RED + "Hai sicuramente il diabete")
        reset_color()
        self.declare(Fact(diabete_tutti_sintomi = "si"))


    @Rule(Fact(prescrizione_esami_sangue="si"))
    def prescription_1(self):
        print(Fore.RED + "Dovresti fare gli esami per misurare la glicemia nel sangue!")
        reset_color()

        self._prototype_lab_booking("gli esami della glicemia nel sangue",self.lab_glucose_analysis)

    @Rule(Fact(glicemia_normale="si"))
    def normal_blood_glucose(self):
        print(Fore.GREEN + "La glicemia e' nella norma")
        reset_color()

    @Rule(NOT(AND(Fact(sete="si"),Fact(stanchezza="si"),Fact(perdita_peso="si"),Fact(prurito="si"),Fact(vista_offuscata="si"),Fact(bevande_zuccherate="si"),Fact(fame_costante="si"),Fact(poliuria="si"))))
    def not_symptoms(self):

        if self.number_prints == 0 and self.flag_no_symptoms == 1:

            print(Fore.GREEN + "Non hai alcun sintomo del diabete")
            self.declare(Fact(niente_sintomi="si"))
            reset_color()
            self.number_prints = self.number_prints + 1

    @Rule(NOT(OR(Fact(diabete_tutti_sintomi = "si"),Fact(tutti_sintomi="si"))))
    def intermediate_case(self):

        if self.flag_no_symptoms != 1:

            print(Fore.YELLOW + "Potresti avere il diabete, rivolgiti ad un medico")
            self.declare(Fact(diagnosi_diabete_incerta = "si"))
            reset_color()


def main_agent():
    expert_agent = esperto_diabete()
    expert_agent.reset()
    expert_agent.run()
    expert_agent.print_facts()


def main_ontology():
    do = diabetes_ontology()

    do.get_symptoms_descriptions()
    symptoms, keys_symptoms = do.print_symptoms()

    print("\nSeleziona il sintomo di cui vuoi conosere la descrizione, inserisci il numero del sintomo")
    symptom_number = int(input())

    while symptom_number not in symptoms.keys():
        print("\nSeleziona il sintomo di cui vuoi conosere la descrizione, inserisci il numero del sintomo")
        symptom_number = int(input())
            
    print("Sintomo: %s, descrizione: %s"%(keys_symptoms[symptom_number]," ".join(symptoms[symptom_number])))



if __name__ == '__main__':

    exit_program = False

    print("Benvenuto in Esperto Diabete, un sistema esperto per la diagnosi del diabete")
    while exit_program == False:

        print("----------->MENU<-----------\n[1] Mostra i possibili sintomi del diabete\n[2] Esegui una diagnosi\n[3] Esci")
        user_choose = None

        try:
           # print("/n") scrivere per input
            user_choose = int(input())
        
        except ValueError:
            exit_program = True

        if user_choose == 1:
            main_ontology()

        elif user_choose == 2:
            main_agent()
        
        else:
            print("Uscita dal programma...")
            exit_program = True
        
        print("\n\n")