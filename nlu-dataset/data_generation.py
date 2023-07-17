import pandas as pd
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
   AutoTokenizer, 
   AutoModelForSeq2SeqLM
)
import os

def augment_templates_with_entities(templates, entities, pattern):
   new_examples = []
   for template in tqdm(templates):
      if pattern in template:
         for entity in entities:
            example = template.replace(pattern, entity, 1)
            if pattern in example:
               for second_entity in entities:
                  if second_entity != entity:
                     second_example = example.replace(pattern, second_entity, 1)
                     new_examples.append(second_example)
            else:
               new_examples.append(example)
      else:
         new_examples.append(template)
   
   return new_examples


class Paraphraser:
   def __init__(self, tokenizer, model, device):
      self.device = device
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
      self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
   
   def __call__(
      self,
      phrase,
      num_beams=5,
      num_beam_groups=5,
      num_return_sequences=5,
      repetition_penalty=10.0,
      diversity_penalty=3.0,
      no_repeat_ngram_size=2,
      temperature=0.7,
      max_length=512):

      input_ids = self.tokenizer(
        f'paraphrase: {phrase}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
      ).input_ids.to(self.device)

      outputs = self.model.generate(
         input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
         num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
         num_beams=num_beams, num_beam_groups=num_beam_groups,
         max_length=max_length, diversity_penalty=diversity_penalty)
      res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

      return res


paraphraser = Paraphraser("humarin/chatgpt_paraphraser_on_T5_base", "humarin/chatgpt_paraphraser_on_T5_base", "cuda")


def augment_once(examples, paraphraser):
   new_set_of_examples = []
   new_set_of_examples += examples
   for example in tqdm(examples):
      paraphrased_examples = paraphraser(example)
      for paraphrased_example in paraphrased_examples:
         if paraphrased_example not in new_set_of_examples:
            new_set_of_examples.append(paraphrased_example)
   return new_set_of_examples


def augment_twice(examples, paraphraser):
   new_set_of_examples = []
   new_set_of_examples += examples
   for example in tqdm(examples):
      paraphrased_examples = paraphraser(example)
      for paraphrased_example in paraphrased_examples:
         if paraphrased_example not in new_set_of_examples:
            new_set_of_examples.append(paraphrased_example)
            new_paraphrased_examples = paraphraser(paraphrased_example, num_beams=10, num_beam_groups=2, num_return_sequences=10)
            for new_paraphrased_example in new_paraphrased_examples:
               if new_paraphrased_example not in new_set_of_examples:
                  new_set_of_examples.append(new_paraphrased_example)
   return new_set_of_examples

entity_names_to_files = {
   "weapons":"weapons.json",
   "food":"food.json",
   "cloth_armor":"cloth_armor.json",
   "alchemy":"alchemy.json",
   "places":"places.json",
   "attacking_verbs":"attacking_verbs.json",
   "creatures_for_attacking":"creatures_for_attacking.json",
   "protecting_verbs":"protecting_verbs.json",
   "creatures_for_protecting":"creatures_for_protecting.json",
   "delivering_verbs":"delivering_verbs.json",
   "destination_creatures":"destination_creatures.json",
   "deliver_message_verbs":"deliver_message_verbs.json",
   "message_destination_creatures":"message_destination_creatures.json",
   "random_words":"random_words.json",
   "greeting_creatures":"greeting_creatures.json",
   "good_farewell_creatures":"good_farewell_creatures.json",
   "bad_farewell_creatures":"bad_farewell_creatures.json",
   "people":"people.json",
   "countries":"countries.json",
   "cities":"cities.json",
   "rivers":"rivers.json",
   "join_verbs":"join_verbs.json",
   "follow_creatures":"follow_creatures.json",
   "move_destination":"move_destination.json",
   "job_words":"job_words.json",
   "job_places":"job_places.json",
   "questing_words":"questing_words.json",
   "creatures_for_threatening":"creatures_for_threatening.json"
}

template_names_to_files = {
   "action_to_someone":"action_to_someone.json",
   "action_on_someone":"action_on_someone.json",
   "single_getting":"single_getting.json",
   "double_getting":"double_getting.json",
   "single_giving":"single_giving.json",
   "double_giving":"double_giving.json",
   "general_exchanging_examples":"general_exchanging_examples.json",
   "greeting":"greeting.json",
   "good_farewell":"good_farewell.json",
   "bad_farewell":"bad_farewell.json",
   "general_knowledge":"general_knowledge.json",
   "people_knowledge":"people_knowledge.json",
   "join":"join.json",
   "follow":"follow.json",
   "move":"move.json",
   "recieve_quest":"recieve_quest.json",
   "complete_quest":"complete_quest.json",
   "threatening":"threatening.json"
}

entities = {}
templates = {}

for key in entity_names_to_files.keys():
   with open(os.join("entities/",entity_names_to_files[key])) as file:
      entities[key] = json.load(file)

for key in template_names_to_files.keys():
   with open(os.path.join("templates/", template_names_to_files[key])) as file:
      templates[key] = json.load(file)

entities["exchange_entities"] = entities["weapons"] + entities["food"] + entities["cloth_armor"] + entities["alchemy"]


def generate_exchange_examples():
   getting_examples = []
   getting_examples += augment_templates_with_entities(templates["single_getting"], entities["exchange_entities"], "[ITEM]")
   getting_examples += augment_templates_with_entities(templates["double_getting"], entities["weapons"], "[ITEM]")
   getting_examples += augment_templates_with_entities(templates["double_getting"], entities["food"], "[ITEM]")
   getting_examples += augment_templates_with_entities(templates["double_getting"], entities["cloth_armor"], "[ITEM]")
   getting_examples += augment_templates_with_entities(templates["double_getting"], entities["alchemy"], "[ITEM]")
   random.shuffle(getting_examples)

   giving_examples = []
   giving_examples += augment_templates_with_entities(templates["single_giving"], entities["exchange_entities"], "[ITEM]")
   giving_examples += augment_templates_with_entities(templates["double_giving"], entities["weapons"], "[ITEM]")
   giving_examples += augment_templates_with_entities(templates["double_giving"], entities["cloth_armor"], "[ITEM]")
   giving_examples += augment_templates_with_entities(templates["double_giving"], entities["food"], "[ITEM]")
   giving_examples += augment_templates_with_entities(templates["double_giving"], entities["alchemy"], "[ITEM]")
   random.shuffle(giving_examples)

   new_set_of_general_exchanging_examples = augment_twice(templates["general_exchanging_examples"], paraphraser)

   random.shuffle(new_set_of_general_exchanging_examples)

   exchanging_examples = []
   exchanging_examples += random.sample(getting_examples, k=(int)(0.05 * len(getting_examples)))
   exchanging_examples += random.sample(giving_examples, k=(int)(0.05 * len(giving_examples)))
   exchanging_examples += new_set_of_general_exchanging_examples
   random.shuffle(exchanging_examples)

   labels = ["Exchange" for i in range(len(exchanging_examples))]

   exchanging_dict = {"examples":exchanging_examples, "labels": labels}

   data_df = pd.DataFrame(exchanging_dict)

   return data_df


def generate_attack_examples():
   attacking_templates = augment_templates_with_entities(templates["action_on_someone"], entities["attacking_verbs"], "[ACTION]")

   attacking_examples = augment_templates_with_entities(attacking_templates,entities["creatures_for_attacking"],"[CREATURE]")

   attacking_examples = augment_once(attacking_examples, paraphraser)

   random.shuffle(attacking_examples)

   more_clear_attacking_examples = random.sample(attacking_examples,k=(int)(0.5 * len(attacking_examples)))

   attack_labels = ["Attack" for i in range(len(more_clear_attacking_examples))]

   attack_dict = {"Examples": more_clear_attacking_examples, "labels": attack_labels}

   attack_df = pd.DataFrame(attack_dict)

   return attack_df


def generate_protect_examples():
   protecting_templates = augment_templates_with_entities(templates["action_on_someone"],entities["protecting_verbs"],"[ACTION]")

   protecting_examples = augment_templates_with_entities(protecting_templates,entities["creatures_for_protecting"],"[CREATURE]")

   more_protecting_examples = augment_once(protecting_examples, paraphraser)

   random.shuffle(more_protecting_examples)

   some_protecting_examples = random.sample(more_protecting_examples,k=(int)(0.55 * len(more_protecting_examples)))

   protecting_labels = ["Protect" for i in range(len(some_protecting_examples))]

   protecting_dict = {"examples":some_protecting_examples, "labels":protecting_labels}

   protecting_df = pd.DataFrame(protecting_dict)

   return protecting_df


def generate_deliver_examples():

   delivering_items = entities["weapons"] + entities["alchemy"] + entities["cloth_armor"]

   delivering_actions = augment_templates_with_entities(templates["delivering_verbs"], delivering_items,"[ITEM]")

   delivering_templates = augment_templates_with_entities(templates["action_to_someone"], delivering_actions, "[ACTION]")

   delivering_examples = augment_templates_with_entities(delivering_templates,entities["destination_creatures"],"[DESTINATION]")
   random.shuffle(delivering_examples)

   length_of_paraphrased = (int)(0.1 * len(delivering_examples))
   paraphrased_delivering_examples = augment_once(delivering_examples[:length_of_paraphrased], paraphraser)

   random.shuffle(paraphrased_delivering_examples)

   final_delivering_examples = random.sample(paraphrased_delivering_examples, k=(int)(0.07 * len(paraphrased_delivering_examples))) + random.sample(delivering_examples[length_of_paraphrased:], k=(int)(0.05 * len(delivering_examples[length_of_paraphrased:])))

   delivering_labels = ["Deliver" for i in range(len(final_delivering_examples))]

   delivering_dict = {"examples": final_delivering_examples, "labels": delivering_labels}

   delivering_df = pd.DataFrame(delivering_dict)

   return delivering_df


def generate_message_examples():
   message_templates = augment_templates_with_entities(templates["action_to_someone"], entities["deliver_message_verb"], "[ACTION]")

   message_examples = augment_templates_with_entities(message_templates, entities["message_destination_creatures"], "[DESTINATION]")

   paraphrased_message_examples = augment_once(message_examples, paraphraser)

   random.shuffle(paraphrased_message_examples)
   more_clear_message_examples = random.sample(paraphrased_message_examples, k=(int)(0.8 * len(paraphrased_message_examples)))

   message_labels = ["Message" for i in range(len(more_clear_message_examples))]

   message_dict = {"examples":more_clear_message_examples, "labels":message_labels}
   message_df = pd.DataFrame(message_dict)

   return message_df


def generate_drival_examples():
   drival_examples = []
   for i in range(15000):
      num_words = random.randint(1, 25)
      words_subset = random.sample(entities["random_words"],k=num_words)
      example = ""
      for word in words_subset:
         example += word
      drival_examples.append(example)

   random.shuffle(drival_examples)
   drival_subset_examples = random.sample(drival_examples, k=(int)(0.75 * len(drival_examples)))

   drival_labels = ["Drival" for i in range(len(drival_subset_examples))]

   drival_dict = {"examples":drival_subset_examples,"labels": drival_labels}
   drival_df = pd.DataFrame(drival_dict)

   return drival_df


def generate_greeting_examples():
   greeting_examples = augment_templates_with_entities(templates["greeting"],entities["greeting_creatures"],"[CREATURE]")

   paraphrased_greeting_examples = augment_twice(greeting_examples, paraphraser)

   random.shuffle(paraphrased_greeting_examples)
   more_clear_paraphrased_greeting_examples = random.sample(paraphrased_greeting_examples, k=(int)(0.8 * len(paraphrased_greeting_examples)))

   greeting_labels = ["Greeting" for i in range(len(more_clear_paraphrased_greeting_examples))]

   greeting_dict = {"examples": more_clear_paraphrased_greeting_examples, "labels":greeting_labels }
   greeting_df = pd.DataFrame(greeting_dict)

   return greeting_df


def generate_farewell_examples():
   good_farewell_examples = augment_templates_with_entities(templates["good_farewell"], entities["good_farewell_creatures"], "[CREATURE]")
   good_farewell_examples = augment_once(good_farewell_examples, paraphraser)

   bad_farewell_examples = augment_templates_with_entities(templates["bad_farewell"], entities["bad_farewell_creatures"], "[CREATURE]")
   bad_farewell_examples = augment_once(bad_farewell_examples, paraphraser)

   random.shuffle(bad_farewell_examples)

   random.shuffle(good_farewell_examples)

   farewell_examples = good_farewell_examples + bad_farewell_examples
   random.shuffle(farewell_examples)

   farewell_labels = ["Farewell" for i in range(len(farewell_examples))]

   farewell_dict = {
      "examples": farewell_examples,
      "labels": farewell_labels
   }
   farewell_df = pd.DataFrame(farewell_dict)

   return farewell_df


def generate_knowledge_examples():
   facts = entities["rivers"] + entities["people"] + entities["cities"] + entities["places"] + entities["countries"]

   knowledge_examples = []
   knowledge_examples += augment_templates_with_entities(templates["general_knowledge"], facts, "[FACT]")
   knowledge_examples += augment_templates_with_entities(templates["people_knowledge"], entities["people"], "[CREATURE]")
   random.shuffle(knowledge_examples)

   more_knowledge_examples = augment_once(knowledge_examples, paraphraser)
   random.shuffle(more_knowledge_examples)

   knowledge_labels = ["Knowledge" for i in range(len(more_knowledge_examples))]

   knowledge_dict = {
      "examples": more_knowledge_examples,
      "labels": knowledge_labels
   }
   knowledge_df = pd.DataFrame(knowledge_dict)

   return knowledge_df


def generate_join_examples():
   join_examples = augment_templates_with_entities(templates["join"], entities["join_verbs"], "[JOIN]")
   random.shuffle(join_examples)

   more_join_examples = augment_twice(join_examples, paraphraser)
   random.shuffle(more_join_examples)

   join_labels = ["Join" for i in range(len(more_join_examples))]

   join_dict = {
      "examples": more_join_examples,
      "labels": join_labels
   }
   join_df = pd.DataFrame(join_dict)

   return join_df


def generate_follow_examples():
   follow_examples = augment_templates_with_entities(templates["follow"], entities["follow_creatures"], "[CREATURE]")
   random.shuffle(follow_examples)

   more_follow_examples = augment_once(follow_examples, paraphraser)
   random.shuffle(more_follow_examples)

   follow_labels = ["Follow" for i in range(len(more_follow_examples))]

   follow_dict = {
      "examples": more_follow_examples,
      "labels": follow_labels
   }
   follow_df = pd.DataFrame(follow_dict)

   return follow_df


def generate_move_templates():
   move_examples = augment_templates_with_entities(templates["move"], entities["move_destination"], "[DESTINATION]")
   random.shuffle(move_examples)

   more_move_examples = augment_once(move_examples, paraphraser)
   random.shuffle(more_move_examples),

   move_labels = ["Move" for i in range(len(more_move_examples))]

   move_dict = {
      "examples": more_move_examples,
      "labels": move_labels
   }
   move_df = pd.DataFrame(move_dict)

   return move_df


def generate_recieve_quest_examples():
   recieving_quest_templates = augment_templates_with_entities(templates["recieve_quest"], entities["job_words"], "[JOB]")

   recieve_quest_examples = augment_templates_with_entities(recieving_quest_templates, entities["job_places"], "[PLACE]")
   random.shuffle(recieve_quest_examples)

   more_recieve_quest_examples = augment_once(recieve_quest_examples, paraphraser)
   random.shuffle(more_recieve_quest_examples)

   recieving_quest_labels = ["Recieve quest" for i in range(len(more_recieve_quest_examples))]

   recieving_quest_dict = {
      "examples": more_recieve_quest_examples,
      "labels": recieving_quest_labels
   }
   recieving_quest_df = pd.DataFrame(recieving_quest_dict)

   return recieving_quest_df


def filter_joke_examples():
   jokes_df = pd.read_csv("shortjokes.csv", sep=";", index_col=False)

   jokes_examples = jokes_df["Joke"].to_list()

   jokes_examples = random.sample(jokes_examples, k=5000)

   jokes_labels = ["Joke" for i in range(len(jokes_examples))]

   jokes_dict = {
      "examples": jokes_examples,
      "labels": jokes_labels
   }
   joke_df = pd.DataFrame(jokes_dict)

   return joke_df


def generate_complete_quest_examples():
   complete_quest_subtemplates = augment_templates_with_entities(templates["complete_quest"], entities["questing_words"], "[JOB]")

   complete_quest_examples = augment_templates_with_entities(complete_quest_subtemplates, entities["places"], "[PLACE]")

   more_complete_quest_examples = augment_once(complete_quest_examples, paraphraser)
   random.shuffle(more_complete_quest_examples)

   complete_quest_labels = ["Complete quest" for i in range(len(more_complete_quest_examples))]

   complete_quest_dict = {
      "examples":more_complete_quest_examples,
      "labels":complete_quest_labels
   }
   complete_quest_df = pd.DataFrame(complete_quest_dict)

   return complete_quest_df


def generate_threatening_examples():
   threatening_examples = augment_templates_with_entities(templates["threatening"], entities["creatures_for_threatening"], "[CREATURE]")
   random.shuffle(threatening_examples)

   paraphrased_threatening_examples = augment_once(threatening_examples[:200], paraphraser)
   random.shuffle(paraphrased_threatening_examples)

   more_threatening_examples = paraphrased_threatening_examples + threatening_examples[200:]

   threatening_labels = ["Threat" for i in range(len(more_threatening_examples))]

   threatening_dict = {
      "examples":more_threatening_examples,
      "labels":threatening_labels
   }
   threatening_df = pd.DataFrame(threatening_dict)

   return threatening_df


def filter_general_examples():
   dataset = load_dataset("silicone", "dyda_da")

   inform_examples = []
   for example in dataset["train"]:
      if example["Label"] == 2:
         inform_examples.append(example["Utterance"])

   general_examples = random.sample(inform_examples, k=int(0.3*len(inform_examples)))

   general_labels = ["General" for i in range(len(general_examples))]

   general_dict = {
      "examples": general_examples,
      "labels": general_labels
   }
   general_df = pd.DataFrame(general_dict)

   return general_df


exchange_df = generate_exchange_examples()
attack_df = generate_attack_examples()
protect_df = generate_protect_examples()
deliver_df = generate_deliver_examples()
message_df = generate_message_examples()
drival_df = generate_drival_examples()
greeting_df = generate_greeting_examples()
farewell_df = generate_farewell_examples()
knowledge_examples = generate_knowledge_examples()
join_df = generate_join_examples()
follow_df = generate_follow_examples()
move_df = generate_move_templates()
recieve_quest_df = generate_recieve_quest_examples()
joke_df = filter_joke_examples()
complete_quest_df = generate_complete_quest_examples()
threatening_df = generate_threatening_examples()
general_df = filter_general_examples()

exchange_df.to_csv("exchange_examples.csv", index=False, sep=";")
attack_df.to_csv("attacking_examples.csv", index=False, sep=";")
protect_df.to_csv("protecting_examples.csv", index=False, sep=";")
deliver_df.to_csv("delivering_examples.csv", index=False, sep=";")
drival_df.to_csv("drival_examples.csv", index=False, sep=";")
greeting_df.to_csv("greeting_examples.csv", index=False, sep=";")
farewell_df.to_csv("farewell_examples.csv", index=False, sep=";")
knowledge_examples.to_csv("knowledge_examples.csv", index=False, sep=";")
join_df.to_csv("joining_examples.csv", index=False, sep=";")
follow_df.to_csv("following_examples.csv", index=False, sep=";")
move_df.to_csv("moving_examples.csv", index=False, sep=";")
recieve_quest_df.to_csv("recieving_quest_examples.csv", index=False, sep=";")
joke_df.to_csv("joke_examples.csv", index=False, sep=";")
complete_quest_df.to_csv("completing_examples.csv", index=False, sep=";")
threatening_df.to_csv("threatening_examples.csv", index=False, sep=";")
general_df.to_csv("general_examples.csv", index=False, sep=";")