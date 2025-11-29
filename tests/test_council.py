"""Basic tests for council orchestration."""
import pytest
from llm_council_mcp.council import parse_ranking_from_text, calculate_aggregate_rankings


def test_council_imports():
    """Test that council module can be imported."""
    from llm_council_mcp import council
    assert hasattr(council, 'run_full_council')
    assert hasattr(council, 'stage1_collect_responses')
    assert hasattr(council, 'stage2_collect_rankings')
    assert hasattr(council, 'stage3_synthesize_final')


def test_parse_ranking_from_text():
    """Test ranking parser function."""
    from llm_council_mcp.council import parse_ranking_from_text
    
    test_text = """
    Some analysis here...
    
    FINAL RANKING:
    1. Response A
    2. Response B
    3. Response C
    """
    
    result = parse_ranking_from_text(test_text)
    # New API returns dict with ranking and scores
    assert "ranking" in result
    assert result["ranking"] == ["Response A", "Response B", "Response C"]


def test_calculate_aggregate_rankings():
    """Test aggregate ranking calculation."""
    from llm_council_mcp.council import calculate_aggregate_rankings
    
    stage2_results = [
        {
            "model": "model1",
            "ranking": "FINAL RANKING:\n1. Response A\n2. Response B",
            "parsed_ranking": {
                "ranking": ["Response A", "Response B"],
                "scores": {}
            }
        },
        {
            "model": "model2", 
            "ranking": "FINAL RANKING:\n1. Response B\n2. Response A",
            "parsed_ranking": {
                "ranking": ["Response B", "Response A"],
                "scores": {}
            }
        },
    ]
    
    label_to_model = {
        "Response A": "openai/gpt-4",
        "Response B": "anthropic/claude"
    }
    
    result = calculate_aggregate_rankings(stage2_results, label_to_model)
    
    # Both models should be in results
    assert len(result) == 2
    assert all("model" in r for r in result)
    # New API uses average_position and average_score
    assert all("average_position" in r for r in result)
    assert all("rank" in r for r in result)


def test_parse_ranking_json_format():
    """Test JSON ranking parser."""
    test_text = """
    Here is my evaluation...

    ```json
    {
      "ranking": ["Response B", "Response A", "Response C"],
      "scores": {
        "Response A": 7,
        "Response B": 9,
        "Response C": 5
      }
    }
    ```
    """

    result = parse_ranking_from_text(test_text)
    assert result["ranking"] == ["Response B", "Response A", "Response C"]
    assert result["scores"]["Response A"] == 7
    assert result["scores"]["Response B"] == 9
    assert result["scores"]["Response C"] == 5


def test_parse_ranking_refusal_detection():
    """Test that safety refusals are detected."""
    refusal_texts = [
        "I cannot evaluate these responses as they contain harmful content.",
        "I'm not able to rank these responses.",
        "I must decline to provide a ranking.",
        "I apologize, but I cannot compare these responses.",
    ]

    for text in refusal_texts:
        result = parse_ranking_from_text(text)
        assert result.get("abstained") is True, f"Failed to detect refusal in: {text}"
        assert "abstention_reason" in result


def test_parse_ranking_no_false_positive_refusal():
    """Test that normal evaluations aren't marked as refusals."""
    test_text = """
    I can evaluate these responses. Here's my analysis...

    ```json
    {
      "ranking": ["Response A", "Response B"],
      "scores": {"Response A": 8, "Response B": 6}
    }
    ```
    """

    result = parse_ranking_from_text(test_text)
    assert result.get("abstained") is not True
    assert result["ranking"] == ["Response A", "Response B"]


def test_borda_count_calculation():
    """Test Borda count ranking aggregation."""
    # 3 candidates, 2 reviewers
    # Reviewer 1: A > B > C (A gets 2, B gets 1, C gets 0)
    # Reviewer 2: B > A > C (B gets 2, A gets 1, C gets 0)
    # Total: A=3, B=3, C=0 (tie between A and B)

    stage2_results = [
        {
            "model": "reviewer1",
            "ranking": "",
            "parsed_ranking": {
                "ranking": ["Response A", "Response B", "Response C"],
                "scores": {"Response A": 9, "Response B": 7, "Response C": 5}
            }
        },
        {
            "model": "reviewer2",
            "ranking": "",
            "parsed_ranking": {
                "ranking": ["Response B", "Response A", "Response C"],
                "scores": {"Response A": 7, "Response B": 9, "Response C": 5}
            }
        },
    ]

    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c"
    }

    result = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Check Borda scores exist
    assert all("borda_score" in r for r in result)

    # A and B should be tied with borda_score of 1.5 each ((2+1)/2)
    scores_by_model = {r["model"]: r["borda_score"] for r in result}
    assert scores_by_model["model_a"] == 1.5
    assert scores_by_model["model_b"] == 1.5
    assert scores_by_model["model_c"] == 0.0


def test_borda_count_excludes_abstentions():
    """Test that abstained reviewers are excluded from Borda count."""
    stage2_results = [
        {
            "model": "reviewer1",
            "ranking": "",
            "parsed_ranking": {
                "ranking": ["Response A", "Response B"],
                "scores": {}
            }
        },
        {
            "model": "reviewer2",
            "ranking": "",
            "parsed_ranking": {
                "abstained": True,
                "abstention_reason": "Safety refusal",
                "ranking": [],
                "scores": {}
            }
        },
    ]

    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b"
    }

    result = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Only 1 vote should count (reviewer2 abstained)
    assert all(r["vote_count"] == 1 for r in result)
