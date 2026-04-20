import os
import json
import requests
import csv
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from typing import List, Optional
import time
from copy import copy, deepcopy
from dashscope import Generation

class LLM_Proxy():
    """
    Universal LLM Request Proxy
    Supports multiple LLMs (OpenAI, Qwen, DeepSeek, Claude, Gemini, Kimi, etc.)
    Provides unified interface and error handling
    """

    def llm_request(self, system_prompt, user_prompt, model_name, temperature=0.7):
        """
        Unified LLM call entry
        Routes requests to corresponding LLM based on model name
        """
        if model_name == "qwen3-max":
            return self.llm_request_qwen(system_prompt, user_prompt, model='qwen3-max', temperature=temperature, enable_thinking=False)
        elif model_name == "qwen3-32b":
            return self.llm_request_qwen(system_prompt, user_prompt, model='qwen3-32b', temperature=temperature, enable_thinking=False)
        
    # you can complete the class here
    

    


class LLM_Individual_Generation(LLM_Proxy):
    """
    Single-sample LLM generation with automatic error retry
    Must inherit and override: get_prompt(), check_generation()
    Retries automatically on API error or invalid output
    """
    def __init__(
        self,
        save_dir: str,
        err_file_name: str = 'err.csv',
        err_log_file_name: str = 'err.log',
        err_generation_file_name: str = 'err_generation.csv',
        prompt_log_name: str = 'prompt.log',
        model_name: str = 'gpt-4.1',
    ):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Log and result paths
        self.err_path = os.path.join(save_dir, err_file_name)
        self.err_log_path = os.path.join(save_dir, err_log_file_name)
        self.err_generation_path = os.path.join(save_dir, err_generation_file_name)
        self.prompt_log_path = os.path.join(save_dir, prompt_log_name)
        self.model_name = model_name
        self.tolerable_error_type_list = []

    def generate_raw(self, system_prompt, user_prompt, sleep_time=None, retry_num=10, record_prompt=False):
        """
        Direct raw LLM call for batch generation
        No validation check — returns original response
        """
        success = False
        if record_prompt:
            self.record_prompt_log("batch_request", system_prompt, user_prompt, self.prompt_log_path)

        cnt_err_num = 0
        api_error_type_list = ["rate limit", "Connection aborted.", "HTTPSConnectionPool"]
        
        while not success:
            success, output = self.llm_request(system_prompt, user_prompt, self.model_name)
            if success:
                return self.generation_postprocess(output)
            else:
                # Log API errors
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                with open(self.err_log_path, 'a') as err_file:
                    err_file.write("{} | batch_request | {}\n".format(time_str, output))
                
                # Check if error is retryable
                cnt_error = True
                for api_err_type_substring in api_error_type_list:
                    if api_err_type_substring in str(output):
                        cnt_error = False
                        break
                if cnt_error:
                    cnt_err_num += 1
                    if cnt_err_num >= retry_num:
                        with open(self.err_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(["batch_request", output])
                        raise RuntimeError(f"Batch LLM call failed: {output}")

            if sleep_time:
                time.sleep(sleep_time)

    def get_prompt(self, sample_id):
        """
        OVERRIDE THIS METHOD
        Generate system and user prompt for given sample_id
        """
        system_prompt = 'You are a helpful assistant.'
        user_prompt = 'hello'
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        """
        OVERRIDE THIS METHOD
        Validate if generation result is valid
        Return (is_valid, error_info)
        """
        error_info = None
        return True, error_info


    def generation_postprocess(self, raw_generation_str):
        """
        Clean markdown/json formatting from LLM output
        Remove ```json, ```, triple quotes
        """
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str


    def record_prompt_log(self, sample_id, system_prompt, user_prompt, prompt_log_path):
        """Save prompt to log file for debugging"""
        dump_str = '---------- {} ----------\n【system prompt】\n{}\n【user prompt】\n{}\n\n'.format(sample_id, system_prompt, user_prompt)
        with open(prompt_log_path, 'a') as prompt_log_file:
            prompt_log_file.write(dump_str)


    def generate(self, sample_id, sleep_time=None, retry_num=10, record_prompt=False):
        """
        Generate single sample with full retry & validation
        Return (success_flag, final_output)
        """
        success = False
        system_prompt, user_prompt = self.get_prompt(sample_id)
        if record_prompt:
            self.record_prompt_log(sample_id, system_prompt, user_prompt, self.prompt_log_path)

        cnt_err_num = 0
        api_error_type_list = ["rate limit", "Connection aborted.", "HTTPSConnectionPool"]
        
        while not success:
            success, output = self.llm_request(system_prompt, user_prompt, self.model_name)
            if success:
                # Check if output meets requirements
                success, error_info = self.check_generation(sample_id, output)
                if success:
                    return success, self.generation_postprocess(output)
                else:
                    # Log invalid generation
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                    with open(self.err_log_path, 'a') as err_file:
                        err_file.write("{} | {} | {}\n".format(time_str, sample_id, error_info))
                    with open(self.err_generation_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([sample_id, output])

                    # Handle unrecoverable errors
                    cnt_error = True
                    for api_err_type_substring in api_error_type_list:
                        if api_err_type_substring in str(output):
                            cnt_error = False
                            break
                    if cnt_error:
                        cnt_err_num += 1
                        if cnt_err_num >= retry_num:
                            if error_info in self.tolerable_error_type_list:
                                with open(self.err_log_path, 'a') as err_file:
                                    err_file.write("{} | Ignore error for {} | type: {}\n".format(time_str, sample_id, error_info))
                                return True, self.generation_postprocess(output)
                            else:
                                with open(self.err_path, 'a') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([sample_id, output])
                                return False, None
            else:
                # Log API call failure
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                with open(self.err_log_path, 'a') as err_file:
                    err_file.write("{} | {} | {}\n".format(time_str, sample_id, output))
                
                cnt_error = True
                for api_err_type_substring in api_error_type_list:
                    if api_err_type_substring in str(output):
                        cnt_error = False
                        break
                if cnt_error:
                    cnt_err_num += 1
                    if cnt_err_num >= retry_num:
                        with open(self.err_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([sample_id, output])
                        return False, None

            if sleep_time:
                time.sleep(sleep_time)


class LLM_Sequential_Generation(LLM_Proxy):
    """
    Sequential batch generation with error handling
    Processes samples one by one, saves to CSV
    Retries failed samples automatically
    """

    def __init__(
        self,
        save_dir: str,
        raw_file_name: str = 'raw.csv',
        err_file_name: str = 'err.csv',
        err_log_file_name: str = 'err.log',
        err_generation_file_name: str = 'err_generation.csv',
        model_name: str = 'qwen_max',
        sample_id_list: Optional[List[str]] = None
    ):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.raw_path = os.path.join(save_dir, raw_file_name)
        self.err_path = os.path.join(save_dir, err_file_name)
        self.err_log_path = os.path.join(save_dir, err_log_file_name)
        self.err_generation_path = os.path.join(save_dir, err_generation_file_name)
        self.model_name = model_name
        self.sample_id_list = sample_id_list
        self.tolerable_error_type_list = []

    def get_prompt(self, sample_id):
        """OVERRIDE: Build prompt for each sample"""
        system_prompt = 'You are a helpful assistant.'
        user_prompt = 'hello'
        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation):
        """OVERRIDE: Validate generation quality"""
        error_info = None
        return True, error_info

    def postprocess_for_iterative_generation(self, sample_id, success, generation):
        """
        For sequential-dependent generation
        Save output for use in future samples
        """
        return

    def sequential_generate(self, sleep_time=None, retry_num=10, continue_generate=False):
        """
        Main sequential generation loop
        Saves results to raw.csv
        Logs errors and invalid outputs
        """
        if not continue_generate:
            # Initialize output files
            with open(self.raw_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'generation'])
            with open(self.err_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'generation'])
            with open(self.err_generation_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'generation'])
            err_log_file = open(self.err_log_path, 'w')
            err_log_file.close()

        for sample_id in tqdm(self.sample_id_list):
            success = False
            system_prompt, user_prompt = self.get_prompt(sample_id)

            cnt_err_num = 0
            api_error_type_list = ["rate limit", "Connection aborted.", "HTTPSConnectionPool"]
            
            while not success:
                success, output = self.llm_request(system_prompt, user_prompt, self.model_name)
                if success:
                    success, error_info = self.check_generation(sample_id, output)
                    if success:
                        # Save valid result
                        with open(self.raw_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([sample_id, output])
                        self.postprocess_for_iterative_generation(sample_id, True, output)
                    else:
                        # Log invalid output
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                        with open(self.err_log_path, 'a') as err_file:
                            err_file.write("{} | {} | {}\n".format(time_str, sample_id, error_info))
                        with open(self.err_generation_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([sample_id, output])

                        # Handle max retries
                        cnt_error = True
                        for api_err_type_substring in api_error_type_list:
                            if api_err_type_substring in str(output):
                                cnt_error = False
                                break
                        if cnt_error:
                            cnt_err_num += 1
                            if cnt_err_num >= retry_num:
                                if error_info in self.tolerable_error_type_list:
                                    with open(self.raw_path, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([sample_id, output])
                                    with open(self.err_log_path, 'a') as err_file:
                                        err_file.write("{} | Ignore error for {} | type: {}\n".format(time_str, sample_id, error_info))
                                    self.postprocess_for_iterative_generation(sample_id, True, output)
                                else:
                                    with open(self.err_path, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([sample_id, output])
                                    with open(self.raw_path, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([sample_id, None])
                                    self.postprocess_for_iterative_generation(sample_id, False, None)
                                break
                else:
                    # Log API error
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                    with open(self.err_log_path, 'a') as err_file:
                        err_file.write("{} | {} | {}\n".format(time_str, sample_id, output))
                    
                    cnt_error = True
                    for api_err_type_substring in api_error_type_list:
                        if api_err_type_substring in str(output):
                            cnt_error = False
                            break
                    if cnt_error:
                        cnt_err_num += 1
                        if cnt_err_num >= retry_num:
                            with open(self.err_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([sample_id, output])
                            with open(self.raw_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([sample_id, None])
                            self.postprocess_for_iterative_generation(sample_id, False, None)
                            break

                if sleep_time:
                    time.sleep(sleep_time)

    def afterwards_iter(self, sleep_time=None):
        """Retry failed samples from err.csv"""
        last_err_df = pd.read_csv(self.err_path, header=None, dtype={0: str})
        last_err_ids = [i for i in list(last_err_df[0])]

        lines = []
        err_idx_dict = {}
        f = open(self.raw_path, "r")
        reader = csv.reader(f)
        line = next(reader)
        lines.append(line)
        row_id = 1
        for line in reader:
            lines.append(line)
            if line[0] in last_err_ids:
                err_idx_dict[line[0]] = row_id
            row_id += 1
        f.close()

        new_err_ids = []
        new_err_info = []
        
        for sample_id in tqdm(last_err_ids):
            system_prompt, user_prompt = self.get_prompt(sample_id)
            success, output = self.llm_request(system_prompt, user_prompt, self.model_name)
            if success:
                lines[err_idx_dict[sample_id]] = [sample_id, output]
            else:
                new_err_ids.append(sample_id)
                new_err_info.append(output)
            if sleep_time:
                time.sleep(sleep_time)
        
        # Write back repaired results
        f = open(self.raw_path, "w")
        writer = csv.writer(f)
        writer.writerows(lines)
        f.close()

        # Save new errors
        new_err_df = pd.DataFrame({'err_ids': new_err_ids, 'err_info': new_err_info})
        new_err_df.to_csv(self.err_path, header=None, index=None)

        print('{} error samples after this iteration: {}'.format(len(new_err_ids), new_err_ids))
        return new_err_ids

    def generation_postprocess(self, raw_generation_str):
        """Strip markdown formatting from LLM output"""
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str


class User_Assistant_Interaction_Generation(LLM_Proxy):
    """
    Multi-turn user-assistant dialogue generation
    Alternates between user and assistant LLMs
    Maintains conversation context
    Must override prompt/check/update methods
    """
    def __init__(
        self,
        save_dir: str,
        raw_file_name: str = 'raw.csv',
        err_file_name: str = 'err.csv',
        err_log_file_name: str = 'err.log',
        err_generation_file_name: str = 'err_generation.csv',
        user_prompt_log_name: str = 'user_llm_prompt.log',
        assistant_prompt_log_name: str = 'assistant_llm_prompt.log',
        user_model_name: str = 'qwen2.5-max',
        assistant_model_name: str = 'qwen_max',
        sample_id_list: Optional[List[str]] = None
    ):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        self.raw_path = os.path.join(save_dir, raw_file_name)
        self.err_path = os.path.join(save_dir, err_file_name)
        self.err_log_path = os.path.join(save_dir, err_log_file_name)
        self.err_generation_path = os.path.join(save_dir, err_generation_file_name)
        self.user_prompt_log_path = os.path.join(save_dir, user_prompt_log_name)
        self.assistant_prompt_log_path = os.path.join(save_dir, assistant_prompt_log_name)
        self.user_model_name = user_model_name
        self.assistant_model_name = assistant_model_name
        self.sample_id_list = sample_id_list
        self.interaction_context_dict = {}
        self.tolerable_error_type_list = []

    def get_user_prompt(self, sample_id):
        """OVERRIDE: Build user prompt"""
        system_prompt = 'You are a user.'
        user_prompt = 'hello'
        return system_prompt, user_prompt

    def get_assistant_prompt(self, sample_id):
        """OVERRIDE: Build assistant prompt"""
        system_prompt = 'You are a helpful assistant.'
        user_prompt = 'hello'
        return system_prompt, user_prompt

    def check_user_generation(self, sample_id, raw_generation):
        """OVERRIDE: Validate user utterance"""
        error_info = None
        return True, error_info
    
    def check_assistant_generation(self, sample_id, raw_generation):
        """OVERRIDE: Validate assistant response"""
        error_info = None
        return True, error_info

    def update_interaction_context(self, role, sample_id, success, generation):
        """OVERRIDE: Update conversation history"""
        return

    def record_prompt_log(self, sample_id, system_prompt, user_prompt, prompt_log_path):
        """Log prompts for debugging"""
        dump_str = '---------- {} ----------\n【system prompt】\n{}\n【user prompt】\n{}\n\n'.format(sample_id, system_prompt, user_prompt)
        with open(prompt_log_path, 'a') as prompt_log_file:
            prompt_log_file.write(dump_str)

    def create_log_files(self, record_prompt):
        """Initialize log and output files"""
        with open(self.raw_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        with open(self.err_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        with open(self.err_generation_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        err_log_file = open(self.err_log_path, 'w')
        err_log_file.close()
        if record_prompt:
            user_prompt_log_file = open(self.user_prompt_log_path, 'w')
            user_prompt_log_file.close()
            assistant_prompt_log_file = open(self.assistant_prompt_log_path, 'w')
            assistant_prompt_log_file.close()

    def assistant_process_before_generation(self, sample_id, record_prompt):
        """Optional pre-processing for assistant"""
        return

    def interaction_generate(self, sleep_time=None, retry_num=10, continue_generate=False, record_prompt=False):
        """
        Main multi-turn dialogue generation loop
        Alternates user → assistant → user → ...
        Saves full conversation to CSV
        """
        if not continue_generate:
            self.create_log_files(record_prompt)

        for sample_id in tqdm(self.sample_id_list):
            for role in ['user', 'assistant']:
                role_sample_id = '{}_{}'.format(sample_id, role)
                success = False
                
                if role == 'user':
                    system_prompt, user_prompt = self.get_user_prompt(sample_id)
                    model_name = copy(self.user_model_name)
                    if record_prompt:
                        self.record_prompt_log(sample_id, system_prompt, user_prompt, self.user_prompt_log_path)
                else:
                    self.assistant_process_before_generation(sample_id, record_prompt)
                    system_prompt, user_prompt = self.get_assistant_prompt(sample_id)
                    model_name = copy(self.assistant_model_name)
                    if record_prompt:
                        self.record_prompt_log(sample_id, system_prompt, user_prompt, self.assistant_prompt_log_path)

                cnt_err_num = 0
                api_error_type_list = ["rate limit", "Connection aborted.", "HTTPSConnectionPool"]
                
                while not success:
                    success, output = self.llm_request(system_prompt, user_prompt, model_name)
                    if success:
                        # Validate generation
                        if role == 'user':
                            success, error_info = self.check_user_generation(sample_id, output)
                        else:
                            success, error_info = self.check_assistant_generation(sample_id, output)
                        
                        if success:
                            # Save valid turn
                            with open(self.raw_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([role_sample_id, output])
                            self.update_interaction_context(role, sample_id, True, output)
                        else:
                            # Log invalid turn
                            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                            with open(self.err_log_path, 'a') as err_file:
                                err_file.write("{} | {} | {}\n".format(time_str, role_sample_id, error_info))
                            with open(self.err_generation_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([role_sample_id, output])

                            # Handle max retries
                            cnt_error = True
                            for api_err_type_substring in api_error_type_list:
                                if api_err_type_substring in str(output):
                                    cnt_error = False
                                    break
                            if cnt_error:
                                cnt_err_num += 1
                                if cnt_err_num >= retry_num:
                                    if error_info in self.tolerable_error_type_list:
                                        with open(self.raw_path, 'a') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([role_sample_id, output])
                                        with open(self.err_log_path, 'a') as err_file:
                                            err_file.write("{} | Ignore error for {} | type: {}\n".format(time_str, role_sample_id, error_info))
                                        self.update_interaction_context(role, sample_id, True, output)
                                    else:
                                        with open(self.err_path, 'a') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([role_sample_id, output])
                                        with open(self.raw_path, 'a') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([role_sample_id, None])
                                        self.update_interaction_context(role, sample_id, False, None)
                                    break
                    
                    else:
                        # Log API error
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                        with open(self.err_log_path, 'a') as err_file:
                            err_file.write("{} | {} | {}\n".format(time_str, role_sample_id, output))
                        
                        cnt_error = True
                        for api_err_type_substring in api_error_type_list:
                            if api_err_type_substring in str(output):
                                cnt_error = False
                                break
                        if cnt_error:
                            cnt_err_num += 1
                            if cnt_err_num >= retry_num:
                                with open(self.err_path, 'a') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([role_sample_id, output])
                                with open(self.raw_path, 'a') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([role_sample_id, None])
                                self.update_interaction_context(role, sample_id, False, None)
                                break
                    
                    if sleep_time:
                        time.sleep(sleep_time)

    def generation_postprocess(self, raw_generation_str):
        """Clean markdown formatting"""
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str