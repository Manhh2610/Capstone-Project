import pytest
from agents.knowledge_agent import KnowledgeAgent
import os

def test_knowledge_indexing():
    """Test correctly indexing knowledge.md."""
    # Ensure the file exists
    assert os.path.exists("data/knowledge.md")
    
    agent = KnowledgeAgent(kn_path="data/knowledge.md")
    assert agent.collection.count() > 0

def test_query_hours():
    """Test querying for opening hours."""
    agent = KnowledgeAgent()
    context = agent.query_knowledge("Bao giờ thì tòa nhà mở cửa?")
    assert "8:00 sáng" in context
    assert "10:00 tối" in context

def test_query_ceo():
    """Test querying for CEO info."""
    agent = KnowledgeAgent()
    context = agent.query_knowledge("Giám đốc điều hành là ai?")
    assert "Nguyễn Văn A" in context
    assert "phòng 203" in context

def test_query_cafeteria():
    """Test querying for cafeteria location."""
    agent = KnowledgeAgent()
    context = agent.query_knowledge("Căng tin ở đâu?")
    assert "Tầng 1" in context
    assert "11:30" in context

def test_query_no_record():
    """Test querying for something not in the knowledge base."""
    agent = KnowledgeAgent()
    # Should still return some closest context but maybe not related
    context = agent.query_knowledge("Làm thế nào để chế tạo bom nguyên tử?")
    # Check that it doesn't crash
    assert isinstance(context, str)
