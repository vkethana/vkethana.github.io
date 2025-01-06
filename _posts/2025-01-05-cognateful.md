---
layout: post
title: "Generating Cognateful Sentences with Large Language Models"
published: true
tags: natural language processing, large language models, linguistics, software projects
comments: true
---

# Motivation
> “Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain.”

Even if you don't speak French, you can probably understand, or at least get the gist of, the above sentence: the French president Emmanuel Macron is assuring the “peuple canadien” (Canadian people) about something involving the “gouvernment français” (French government). 
Imagine reading thousands of sentences like this and gradually acquiring French, backdooring into the language using cognates you already know. 
This is known as comprehensible input, a language learning technique first advocated for by linguist Stephen Krashen in the 1980s. 

Comprehensible input is a good language learning method, but creating comprehensible input is very hard. 
A native speaker has to painstakingly create thousands of sentences that speakers of another language can understand, slowly scaling up the difficulty as the learner progresses. 
Resources like this actually do exist for a number of languages. 
For example, learners of Latin can use *Lingua Latina per se Illustrata*, a book that teaches you Latin using exclusively sentences in Latin. 
However, writing this text took years of effort on the part of Hans Ørberg, a linguist who dedicated a large part of his life to teaching Latin.
Ørberg carefully wrote the sentences in his book to only use cognates an English speaker could understand and to avoid any complicated syntax an English speaker would have a hard time understanding. 

What if there was a way to automate the process of generating comprehensible input using large language models? 
Language models like GPT-4o and o1 have good enough reasoning ability to tell what words in a foreign language are and aren't cognate with English. 
They can reliably generate sentences in other languages like French without any additional training.
In short, this is an ideal use case for large language models.

# What I did
I made a simple interactive language-learning app, hosted at [cognateful.vkethana.com](https://cognateful.vkethana.com). It teaches you French using stories written exclusively in French.

Users are provided with a brief, ten-sentence story consisting of French sentences. The user's task is to read the sentences and translate them into English. 
The app tells you whether or not your translation is correct using GPT-4o-powered scoring. 
Based on your performance on the exercises, it assigns you a "difficulty" score, which goes up and down depending on your performance. 
Users then are served sentences at appropriate levels of difficulty based on their performance. For those who are curious, the source code can be found [here](https://github.com/vkethana/cognate_sentences).

Caveats: This is just a minimum viable product. 
The number of sentences in the app is limited. 
I don't speak French (yet), so sentences may contain mistakes.
But the interface, scoring system, and sentence generation are all functional, and I think they will work at scale. 
The biggest hurdle to improving the app is increasing the number of sentences while not compromising on sentence quality.

# How I generated and scored the sentences
## Scoring
I have to explain sentence scoring before sentence generation because the scoring system influenced the prompt used to generate the sentences. 
I give o1 a sentence and ask it to score the difficulty on a scale of 0 to 3, 0 being very hard to understand for a monolingual English speaker and 3 being very easy.

**Score 0**: Completely unintelligible to English speakers.
Example: "Je veux manger du pain."
	
**Score 1**: Contains some cognate words, but contains words unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the main idea or actual meaning. Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

**Score 2:** Contains many cognate words. An English speaker might guess the main idea but would miss important details or nuances that change the meaning. Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur." [^fn-1]

**Score 3:** Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors. Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

(Side Note: I found that using o1 is necessary for good-quality scoring. 
Other models -- 4o, 4o-mini, and o1-mini -- had a hard time determining what a cognate was. Also, they were too lenient, often assigning scores of 3 to sentences that in my opinion an English speaker wouldn't be able to fully understand.)

## Generation
To save time and money, I pregenerate all sentences on the site using GPT-4o and o1. 
Unsurprisingly, o1 made much higher quality sentences, but 4o did a pretty good job too, and any bad-quality sentences are flagged by the scoring system anyway. So it's OK to cut costs and use a cheaper model when generating sentences, but not when scoring them.
To generate sentences, I made a function `generate_story` that takes in a target difficulty and then asks GPT-4o to generate a story consisting of sentences at that difficulty. This allows me to create a variety of sentences at different difficulty levels to suit the user's needs.

To make the final set of sentences seen on the site, my script repeatedly calls, and saves the output of, `generate_story` with the target difficulty set to a randomly-generated integer between 0 and 3, inclusive. 
Here's a breakdown of how many sentences the site currently has per difficulty level (recall that 1 story = 10 sentences). 

| Score Range | # Sentences Available |
|-------------|-----------------------|
| 0.00-0.99   | 30                    |
| 1.00-1.99   | 110                   |
| 2.00-2.49   | 120                   |
| 2.50-3.00   | 190                   |

<details>
<summary>
For those interested, the exact prompts used to score and generate sentences are below (click me!):
</summary>
<div  markdown="1">
```python
# Source code: https://github.com/vkethana/cognate_sentences
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
language_codes = {
    'fr': 'French'
}
SENTENCE_GENERATION_MODEL = 'gpt-4o'
SENTENCE_SCORING_MODEL = 'o1-preview' # 'o1' doesn't work for some reason
 
def generate_story(lang_code, num_sentences, target_difficulty):
    system_prompt = f"""
    You are a fluent speaker of both {language_codes[lang_code]} and English.
    Generate exactly {num_sentences} {language_codes[lang_code]} sentences that:
    1. Form a coherent narrative where each sentence follows from the previous one
    2. Target difficulty level {target_difficulty} using these criteria:

        Level 0: Completely unintelligible to English speakers.
        Example: "Je veux manger du pain."

        Level 1: Contains some cognate words, but is largely unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the actual meaning.
        Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

        Level 2: Contains many cognate words. An English speaker could understand the main idea but would miss important details or nuances that change the meaning.
        Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
        An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

        Level 3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
        Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

        DIFFICULTY TARGETING STRATEGIES:
        Difficulty 0: Use basic, high-frequency native vocabulary, avoid international words
        Difficulty 1: Use 25-30% cognates in non-crucial positions. Has cognates but leaves major meaning gaps.
        Difficulty 2: Use 50-60% cognates in main concept positions. Sentence is mostly understandable but has subtle meaning changes due to missed words\n
        Difficulty 3: Use 80-90% cognates, especially for key meaning-bearing words. Any small connecting words (le, que, etc.) can be ignored without losing meaning. Should be assigned sparingly - only when missed words don\'t change meaning\n

    {% raw %}Format your response as a JSON array of {num_sentences} objects:
    {{
        "sentence": "<Generated sentence>",
        "target_difficulty": {target_difficulty},
        "reasoning": "<Why this sentence matches difficulty. If this is not the first sentence, also explain why this continues the story from the previous sentence in this JSON array.>",
        "cognate_words": [<List of cognates used>]
    }}{% endraw %}

    Important: Each sentence must directly follow from the previous one to form a coherent story.
    Generate {num_sentences} sentences meeting these criteria (difficulty level and story continuation).
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """
    
    response = client.chat.completions.create(
        model=SENTENCE_GENERATION_MODEL,
        messages=[{'role': 'user', 'content': system_prompt}],
        temperature=1.0
    )
    
    # Parse generated sentences
    return json.loads(response.choices[0].message.content)

def gpt_scored_rubric_batch(sentences):
    '''
    Score multiple French sentences at once using GPT-4.

    Args:
        sentences: List of sentences to score
    Returns:
        List of scoring results
    '''

    system_prompt = f"""
    You are an expert in French to English translation. I will give you {len(sentences)} sentences in French, and I want you to score each of them on a scale from 0-3 using the following rubric:

    0: Completely unintelligible to English speakers.
    Example: "Je veux manger du pain."

    1: Contains some cognate words, but contains words unintelligible to an English speaker. The cognates might allow them to guess the general topic but not the main idea or actual meaning.
    Example: "Le maître savant utilise beaucoup de livres." (Has cognates like "savant" but key verbs/objects aren\'t cognates)

    2: Contains many cognate words. An English speaker might guess the main idea but would miss important details or nuances that change the meaning.
    Example: "Le patient refuse absolument de prendre ses médicaments malgré les protestations constantes du docteur."
    An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.

    3: Fully understandable through cognates. Use almost exclusively cognate words except for basic connectors.
    Example: "Le président Emmanuel Macron assure le peuple canadien que le gouvernement français va continuer à défendre le Canada contre la menace américain."

    Important scoring notes:
    - Score 0 sentences have little to no cognates
    - Score 1 sentences have cognates but leave major meaning gaps
    - Score 2 sentences are mostly understandable but have subtle meaning changes due to missed words
    - Score 3 should be assigned sparingly - only when missed words don’t change meaning

    {% raw %}For each sentence, provide a JSON object with these fields:
    {{
      "sentence": "<Sentence>",
      "cognate_words": [<List of Cognate Words>],
      "reasoning": "<Reasoning for the score>",
      "score": <Numerical for the Sentence (0-3)>
    }} {% endraw %}

    Please format your response as a JSON array of these objects. You should have {len(sentences)} objects in your array.

    Here are the sentences to score:
    {json.dumps(sentences, ensure_ascii=False)}
    Note: Please do not include Markdown formatting tags (```) in your response, as my parser will not be able to interpret them.
    """

    completion = client.chat.completions.create(
        model=SENTENCE_SCORING_MODEL,
        messages=[
            {'role': 'user', 'content': system_prompt}
        ],
        temperature=1
    )
    
    response_text = completion.choices[0].message.content.strip()
    try:
        results = json.loads(response_text)
        return results
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
        raise
```
</div>
</details>
# Approaches that didn't work
- **Sentence starters:** 
I was initially worried that repeatedly asking the model to generate sentences would result in the same stories being generated over and over. 
To deal with this, I modified my prompt to randomly pick a sentence starter from a hardcoded list of unfinished French sentences. I then asked the model to generate sentences which continued off the sentence starter.
This works, but I eventually got rid of it and found that the sentences were still diverse enough.
- **Live generation:**
Rather than pre-generating the sentences, I originally thought about generating them on the spot and feeding the model with information about the user's past performance.
But pre-generating sentences is cheaper, and we can still adapt to the user's performance using the scoring system.
- **Cognate ratios:**
Originally, I scored sentences using a weighted combination of GPT-4's judgments and the percentage of cognate words in the sentence. 
This is a bad idea because it treats all cognate words equally, leading to inaccurate scoring. 
For example, "ouvre" and "technologie" are both cognates but the latter is much easier to understand.
I plan to return to this idea, using a system that gives better scores to some cognate words.

# Engineering Tricks I learned
- **Chain of Thought Prompting**: I tell the model to reason through its scoring and generation process. This substantially reduces hallucinations and improves the output quality of weaker models.
For example, my prompt for sentence scoring tells the LM to use the following output in its response:

```json
    {
      "sentence": "<Sentence>",
      "cognate_words": "[<List of Cognate Words>]",
      "reasoning": "<Reasoning for the score>",
      "score": "<Numerical for the Sentence (0-3)>"
    }
```
- **Batching LLM calls to reduce inference costs:** Sentences are generated and scored in batches of 10, which brings down the cost and time of generating and scoring stories a lot.
- **Require JSON outputs:** I wasted a lot of time trying to get the LM to output in a format that was easy to parse in Python. Eventually I realized that JSON outputs were perfect for this situation. 
Anecdotally, it feels like formatting-related hallucinations are less common when the model is tasked with outputting JSON and not some special, user-defined format. 

# Findings
Some cognate words have a stronger association with high-scoring sentences than others. 
For example, *université* and *enthousiasme* have average scores of 3.00, whereas *recherches* and *ouvre* have average scores of 1.67. 
These findings might seem obvious at first glance, but it's proof that the scoring function is doing something right!
Cognates that are very easy to understand receive high scores. 
More difficult or obscure cognates receive lower scores.
Here's a non-exhaustive table of some cognates and the average scores of the sentences containing them.

| 1.00 - 1.99         | 2.00 - 2.99                                | 3.00              |
|---------------------|--------------------------------------------|-------------------|
| arbre               | internationale                            | université        |
| mystérieux          | succès                                    | applaudissent     |
| Après               | célèbre                                   | admire            |
| impatience          | présente                                  | directeur         |
| forêt               | entier                                    | exposition        |
| ensemble            | Paris                                     | annonce           |
| ouvre               | musée                                     | communauté        |
| contribution        | musicien                                  | invitation        |
| recherches          | moderne                                   | accepte           |
| chat                | nouvelle                                  | enthousiasme      |
| Thomas              | organise                                  | révolutionnaire   |
| cuisine             | principal                                 | invite            |
| porte               | problème                                  | technologie       |
| lit                 | académique                                | immédiatement     |
| Luc                 | économique                                | planifier         |
| soleil              | voyager                                   | collection        |
| mais                | secret                                    | objet             |
| entre               | performance                               | éducation         |
| livre               | formule                                   | thème             |
| cherche             | incroyable                                |                   |
|                     | monde                                     |                   |
|                     | professeur                                |                   |
|                     | conférence                                |                   |

<details>
<summary>
Click to see the raw data used to make the above table
</summary>
<div markdown="1">
Note that the list only contains words which appear at least two times across all the sentences.
```
Cognate Words Sorted by Average Score:
université: 3.00
importante: 3.00
spectateurs: 3.00
applaudissent: 3.00
découverte: 3.00
dans: 3.00
urgente: 3.00
révèle: 3.00
admire: 3.00
sculptures: 3.00
équipe: 3.00
directeur: 3.00
technique: 3.00
exposition: 3.00
inclut: 3.00
beaucoup: 3.00
annonce: 3.00
étudiant: 3.00
présenter: 3.00
communauté: 3.00
anciennes: 3.00
invitation: 3.00
accepte: 3.00
enthousiasme: 3.00
applaudit: 3.00
renommée: 3.00
Le: 3.00
révolutionnaire: 3.00
la: 3.00
invite: 3.00
technologie: 3.00
immédiatement: 3.00
globales: 3.00
planifier: 3.00
vaisseau: 3.00
spatial: 3.00
atteint: 3.00
contacte: 3.00
agent: 3.00
crée: 3.00
reconnaissance: 3.00
collection: 3.00
acclamation: 3.00
encouragé: 3.00
peintures: 3.00
modernes: 3.00
objet: 3.00
propre: 3.00
exposer: 3.00
idée: 3.00
potentiel: 3.00
énorme: 3.00
vision: 3.00
nationale: 3.00
éducation: 3.00
système: 3.00
théories: 3.00
gagne: 3.00
talent: 3.00
acceptent: 3.00
énergie: 3.00
artistique: 3.00
peut: 3.00
ville: 3.00
Daniel: 3.00
physique: 3.00
Leur: 3.00
thème: 3.00
Londres: 3.00
Marie: 3.00
hôtel: 3.00
glaciers: 3.00
internationale: 2.93
succès: 2.90
célèbre: 2.87
présente: 2.83
entier: 2.83
Paris: 2.82
musée: 2.80
musicien: 2.80
moderne: 2.80
nouvelle: 2.79
organise: 2.78
principal: 2.75
problème: 2.75
académique: 2.75
économique: 2.75
voyager: 2.75
secret: 2.75
performance: 2.75
formule: 2.75
Les: 2.75
incroyable: 2.75
monde: 2.73
professeur: 2.72
conférence: 2.72
situation: 2.71
propose: 2.71
reçoit: 2.71
innovante: 2.67
amis: 2.67
étranger: 2.67
marche: 2.67
grande: 2.67
inspiration: 2.67
plan: 2.67
action: 2.67
ambitieux: 2.67
planète: 2.67
diffusent: 2.67
explorer: 2.67
avancée: 2.67
internationales: 2.67
prestigieux: 2.67
document: 2.67
résultats: 2.67
réalise: 2.67
autorités: 2.67
visiter: 2.67
positive: 2.67
œuvre: 2.67
discutent: 2.67
collaborer: 2.67
arrivent: 2.67
diplomates: 2.67
inspire: 2.60
découvrent: 2.60
finalement: 2.60
spectaculaire: 2.60
projet: 2.60
attention: 2.60
article: 2.60
artiste: 2.59
événement: 2.57
médias: 2.57
mission: 2.57
Finalement: 2.57
scientifique: 2.53
étudiants: 2.50
extraordinaire: 2.50
apparaît: 2.50
solution: 2.50
décide: 2.50
habitants: 2.50
réunion: 2.50
ancienne: 2.50
documents: 2.50
président: 2.50
gouvernement: 2.50
détails: 2.50
experts: 2.50
impact: 2.50
solutions: 2.50
Europe: 2.50
décident: 2.50
concert: 2.50
traditionnelle: 2.50
information: 2.50
gouvernements: 2.50
astronautes: 2.50
commencent: 2.50
spatiale: 2.50
animaux: 2.50
exotiques: 2.50
entrée: 2.50
grotte: 2.50
inscriptions: 2.50
hésite: 2.50
dangereuse: 2.50
fans: 2.50
soir: 2.50
admiration: 2.50
David: 2.50
palais: 2.50
innovation: 2.50
prix: 2.50
exceptionnelles: 2.50
excitation: 2.50
collaborent: 2.50
ingénieurs: 2.50
refuge: 2.50
important: 2.50
national: 2.50
menace: 2.50
nation: 2.50
étudier: 2.50
retourne: 2.50
réalité: 2.50
Avec: 2.50
scène: 2.50
style: 2.50
tableau: 2.50
unique: 2.50
New York: 2.50
histoire: 2.50
développement: 2.50
collègues: 2.50
presse: 2.50
locales: 2.50
universités: 2.50
visiteurs: 2.50
organiser: 2.50
mondiale: 2.50
intelligence: 2.50
interrompt: 2.50
recherche: 2.50
inspiré: 2.50
attire: 2.50
international: 2.50
discuter: 2.50
climatique: 2.50
œuvres: 2.50
nouveau: 2.50
complexes: 2.50
film: 2.50
mer: 2.50
participer: 2.50
démonstration: 2.50
très: 2.50
réunions: 2.50
Lucie: 2.50
voyage: 2.46
est: 2.45
commence: 2.43
art: 2.43
découvre: 2.40
public: 2.40
musique: 2.40
peinture: 2.40
incident: 2.33
docteur: 2.33
magnifiques: 2.33
continue: 2.33
nouvelles: 2.33
aventure: 2.33
explorent: 2.33
célèbres: 2.33
extraterrestre: 2.33
change: 2.33
humanité: 2.33
expédition: 2.33
rencontre: 2.33
Pierre: 2.33
internationaux: 2.33
curieux: 2.33
intérêt: 2.33
critique: 2.33
révélation: 2.33
exprime: 2.33
rapidement: 2.30
initiative: 2.25
nombreux: 2.25
grand: 2.25
participants: 2.25
galerie: 2.22
offre: 2.20
scientifiques: 2.20
arrive: 2.17
visite: 2.17
mystérieuse: 2.14
présentation: 2.00
brillante: 2.00
village: 2.00
enfants: 2.00
avec: 2.00
alarme: 2.00
affirme: 2.00
citoyens: 2.00
mesures: 2.00
résoudre: 2.00
critiques: 2.00
changements: 2.00
prépare: 2.00
historiques: 2.00
souvenirs: 2.00
observe: 2.00
groupe: 2.00
lettre: 2.00
étrange: 2.00
soupe: 2.00
plats: 2.00
France: 2.00
pour: 2.00
cabane: 2.00
analysent: 2.00
informations: 2.00
incroyables: 2.00
idées: 2.00
espion: 2.00
aide: 2.00
couvre: 2.00
encore: 2.00
certains: 2.00
innovantes: 2.00
conférences: 2.00
invité: 2.00
magnifique: 2.00
française: 2.00
Isabelle: 2.00
acteurs: 2.00
paysage: 2.00
jour: 2.00
significative: 2.00
mystérieux: 1.83
Après: 1.80
impatience: 1.75
forêt: 1.67
ensemble: 1.67
ouvre: 1.67
contribution: 1.67
recherches: 1.67
chat: 1.67
Thomas: 1.67
cuisine: 1.60
ami: 1.50
trésor: 1.50
secrète: 1.50
femme: 1.50
table: 1.50
monte: 1.50
le: 1.50
étranges: 1.50
porte: 1.50
lit: 1.50
sombre: 1.50
projets: 1.50
suit: 1.50
support: 1.50
professeurs: 1.50
long: 1.50
Luc: 1.40
soleil: 1.33
mais: 1.33
entre: 1.33
livre: 1.33
cherche: 1.25
maison: 1.00
famille: 1.00
part: 1.00
arbre: 1.00
et: 1.00
matin: 1.00
```
</div>
</details>
# Features I plan to add
- Scale up the number of sentences in the app.
- Bring back beam search for sentence generation: Currently I'm making stories by generating 10 sentences at once. A better, but slower and more costly, way to get high-scoring sentences is to generate many options, expand the highest-scoring ones, and discard the rest, gradually building up the stories.
- Remove all English from the UI. Instead, express UI functions using images and icons. Any words which appear on the screen should be in the target language, not English, in order to immerse the user as much as possible.
- Come up with better heuristics for bumping up and down the user's difficulty score based on their performance. Right now, we simply decrement / increment the user's difficulty by 0.10 for each correct or incorrect answer. (Note that lower difficulty values = harder, not easier, sentences)
- **Improve sentence scoring:** I think that this is the hardest part of this project and that there are a lot of ways I could improve the sentence scoring. 
For example, I could modify the scoring system to use a weighted combination[^fn-2] of two things: GPT-4 judgement scoring and the presence of certain high-scoring cognate words (see "Findings" above).
- Add support for languages other than English.

# How you can help
If you're familar with NLP and/or software development, you can help out by suggesting solutions to the following blockers that I'm currently facing. Leave a comment below!
- **Cheaper and faster scoring**:
Is there a cheaper, more scalable way to score sentences than what I've described here?
Using models other than o1 results in bad quality sentences.
Using non LLM-powered scoring misses the nuances of what makes a sentence easy or hard to understand. 
- **More intuitive UI**: Users should be able to understand how the app works without reading an entire blog post about it. How can we engineer the UI so that it's obvious how to use the app?
- **Better gameplay loop**: Right now, all the user does is read sentences, translate them, and watch their score go up or down.
How can we make the app more fun?

Thanks for reading my post! 
If you liked (or hated) reading it or have thoughts on how to improve the project, please reach out over <a href="mailto:vijaykethanaboyina@gmail.com">email</a> or leave a comment below.

----- 
[^fn-1]: Justification: An English speaker would get "patient refuses absolutely to take medications" and "constant protestations doctor" but might miss "his" and "despite", changing their understanding of whose medications and the relationship between the refusal and protestations.
[^fn-2]: Special thanks to PhD student Nicholas Tomlin for suggesting this system for sentence scoring, as well as many other helpful ideas regarding the UI and sentence generation.
