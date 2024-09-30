import tkinter as tk
from tkinter import ttk
import pandas

print(pandas.read_csv("../Dataset/Testing.csv").head(132))

# create a tkinter window
window = tk.Tk()
window.geometry("720x250")
window.title("DigiMed-Python by Md Mahir Alam")

# define the values for the dropdown list
values = ["itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering,chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting","vomiting","burning_micturition","spotting_urination","fatigue","weight_gain","anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness","lethargy","patches_in_throat","irregular_sugar_level","cough","high_fever","sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties","excessive_hunger","extra_marital_contacts","drying_and_tingling_lips","slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck","swelling_joints","movement_stiffness","spinning_movements","loss_of_balance","unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of","urine","continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)","depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body","belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion","receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen","history_of_alcohol_consumption","fluid_overload","blood_in_sputum","prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples","blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze"]

# create a label for the combobox
label = ttk.Label(window, text="Select values:")

# create a combobox
combobox = ttk.Combobox(window, state="readonly")

# create a Listbox widget for the dropdown list
listbox = tk.Listbox(window, selectmode="multiple", exportselection=0)
for value in values:
    listbox.insert(tk.END, value)


# define a function to update the combobox when the user selects or deselects a value
def update_combobox():
    # Get selected values from the Listbox widget
    selected_values = [listbox.get(idx) for idx in listbox.curselection()]

    # Update the combobox with the selected values
    combobox.configure(width=40, height=7)
    combobox.set(", ".join(selected_values))


# bind the update_combobox function to the Listbox widget
listbox.bind("<<ListboxSelect>>", lambda _: update_combobox())

# pack the label, combobox, and Listbox widget
label.pack(side="top", anchor="w", pady=30)
combobox.pack(side="top", pady=30)
listbox.pack(side="top")

# start the main loop
window.mainloop()