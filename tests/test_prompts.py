from components.proposer import LLMProposer


def test_llm_proposer_accepts_literal_prompt():
    literal_prompt = "Describe differences between the groups succinctly."
    proposer = LLMProposer({"prompt": literal_prompt})
    assert proposer.prompt == literal_prompt
