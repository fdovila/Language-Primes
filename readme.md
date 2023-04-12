We propose a hybrid machine learning model that combines human reasoning laws with statistical knowledge from language models (LLMs) for human-level language understanding. The model is represented as follows:

Knowledge graph (G): G(V, E) with vertices V = {LLM1, LLM2, ..., LLMn} and edges E = {e1, e2, ..., em} representing relationships between LLMs.

Semantic memory (semM): A vector semM = {s1, s2, ..., sn} representing the memory strength of each LLM.

Combination rules (combRules): A set of rules, combRules = {r1, r2, ..., rk}, determining how LLMs are combined.

Transformation matrices (transMats): A set of matrices, transMats = {T1, T2, ..., Tk}, defining transformations between combined LLMs.

The model is updated iteratively by receiving inputs (reInp) generated from LLMs using a function generateInputs, which utilizes NLPK and spaCy libraries. The semantic memory and node access probabilities (pTi) are updated with the received inputs:

semM_new = semM + α * (reInp - semM), where α is the learning rate.
pTi = f(freq(ti, reInp)) / Σj f(freq(tj, reInp)), where f is a function mapping frequencies to access probabilities.
The function generateInputs uses LLMs to generate texts and extracts linguistic primes from them. It first calls callOpenAI on each LLM to generate a text, then processes these texts using NLPK and spaCy libraries in the processTextsWithNLPKSpaCy function. The texts are parsed using the nlp model and dependency relations are extracted in dissectInputAndExtractPrimes. Linguistic primes are extracted from dependency relations with the extractLinguisticPrimes function, which groups the dependency relations by type and processes them accordingly in extractPrimesForDependencyType.

We define a function combineNodes to combine LLMs in G according to semM_new and pTi:

G_new = combineNodes(G, semM_new, pTi, combRules, transMats).

The connectivity tensor (CT) is computed using the function conTen, which calculates the tensor product of semM_new and G_new:

CT = conTen(semM_new, G_new) = TensorProduct(semM_new, G_new).

We employ the backpropagation function bkProp to update the model components based on their gradients:

Compute gradients: grad = Grad(CT, {semM_new, pTi, G_new, combRules, transMats}, Constants -> {conTen}).
Update components: semM_new -= ε * grad[1]; pTi -= ε * grad[2]; G_new = combineNodes(G_new, semM_new, pTi, combRules, transMats); combRules -= ε * grad[4]; transMats -= ε * grad[5], where ε is the learning rate.
The natural language summary and evaluation functions are wrapped in a function bprop, which calls the backpropagation function bkProp:

bprop_result = bprop(CT, semM_new, pTi, G_new, combRules, transMats).

The objective is to distill the "semantic and rhetoric" prime-numbers of human language and build constitutional rules for human-level understanding. The rules are incorporated as logic rules, enabling the hybrid ML model to build upon LLMs' statistical knowledge while following human reasoning laws.

FULL CODE:

Here's an example implementation of the extractPrimesForDependencyType function:

extractPrimesForDependencyType[depType_, depRelations_] := Module[{primes},
  primes = Switch[depType,
    "nsubj",
    extractNounPhrases[depRelations],

    "dobj",
    extractVerbPhrases[depRelations],

    _,
    {}
  ];
  primes
];

extractNounPhrases[depRelations_] := Module[{nounPhrases},
  nounPhrases = Table[
    StringJoin[Riffle[token["token"], " "]],
    {token, depRelations}
  ];
  nounPhrases
];

extractVerbPhrases[depRelations_] := Module[{verbPhrases},
  verbPhrases = Table[
    StringJoin[Riffle[{token["head"], token["token"]}, " "]],
    {token, depRelations}
  ];
  verbPhrases
];
