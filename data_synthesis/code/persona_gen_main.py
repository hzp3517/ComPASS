
import user_scenario
import user_utterance
import user_demographic
import read_csv
import tool_invocation_ver1_mul
import json
import os


if __name__ == "__main__":

    code_dir=os.path.dirname(os.path.abspath(__file__))
    root_dir=os.path.dirname(code_dir)
    data_dir=os.path.join(root_dir,f"data")
    demographic_dir=os.path.join(data_dir,f"demographic")


    size = 400
    model = "gpt-5.1"
    demographic_data_save_path=os.path.join(demographic_dir,f"demographic_train_set_300.json")
    scenarios_path=os.path.join(data_dir,f"scenarios",f"scenarios_train_set_300.json")
    utterance_csv_path="/xxx/Empathetic_Interaction/data_synthesis_v2/data/debug/generate_utterance/raw.csv"
    utterance_json_path=os.path.join(data_dir,f"utterance",f"utterance_train_set_300.json")
    tool_invocation_path=os.path.join(data_dir,f"train_set",f"{model}_300.json")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    user_demographic.generate_persona_main(amount = size,save_path=demographic_data_save_path,model= model)

    user_scenario.generate_scenario_main(file_dir = demographic_data_save_path,outdata_dir = scenarios_path,amount=15)

    user_utterance.generate_utterance_main(users_dict_dir = demographic_data_save_path,scenario_dict_dir = scenarios_path,model = model)

    utterance_dict=read_csv.csv_to_utterance_dict(csv_file_path=utterance_csv_path)
    with open(utterance_json_path, "w", encoding="utf-8") as f:
        json.dump(utterance_dict, f, ensure_ascii=False, indent=4)

    tool_invocation_ver1_mul.generate_tool_main(amount=15,user_profile_path=demographic_data_save_path,discourses_path=utterance_json_path,save_path = tool_invocation_path,model = model)    













