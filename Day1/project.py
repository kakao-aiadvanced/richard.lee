import promptbench as pb

# print all supported datasets in promptbench
print('All supported datasets: ')
print(pb.SUPPORTED_DATASETS)

# load a dataset, sst2, for instance.
# if the dataset is not available locally, it will be downloaded automatically.
dataset_name = "gsm8k"
dataset = pb.DatasetLoader.load_dataset(dataset_name)

# print the first 3 examples
print(dataset[:3])

# print all supported models in promptbench
print('All supported models: ')
print(pb.SUPPORTED_MODELS)
print('Supported datasets for each method: ')
print(pb.METHOD_SUPPORT_DATASET)

# load a model, gpt-3.5-turbo, for instance.
# If model is openai/palm, need to provide openai_key/palm_key
# If model is llama, vicuna, need to provide model dir
model = pb.LLMModel(model='gpt-3.5-turbo',
                    api_key = 'xxxxxxxx',
                    max_new_tokens=150)

# load a method, emotion_prompt, for instance.
# https://github.com/microsoft/promptbench/tree/main/promptbench/prompt_engineering
method = pb.PEMethod(method='CoT',#'emotion_prompt',
                        dataset=dataset_name,
                        verbose=True,  # if True, print the detailed prompt and response
                        prompt_id = 1  # for emotion_prompt
                        )

results = method.test(dataset,
                      model,
                      num_samples=3 # if don't set the num_samples, method will use all examples in the dataset
                      )
print(results)