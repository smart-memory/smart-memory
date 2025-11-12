# Conversational AI with SmartMemory

This example demonstrates how to build a conversational AI system that remembers previous conversations and builds context over time using SmartMemory.

## Overview

A memory-enhanced conversational AI can:
- Remember previous conversations across sessions
- Build understanding of user preferences and context
- Provide personalized responses based on history
- Learn from interactions to improve over time

## Basic Implementation

### Simple Memory-Enhanced Chatbot

```python
from smartmemory import SmartMemory
import openai
from datetime import datetime
from typing import List, Dict

class MemoryEnhancedChatbot:
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.memory = SmartMemory()
        openai.api_key = openai_api_key
        
    def chat(self, user_message: str) -> str:
        """Process user message and return AI response with memory context"""
        
        # 1. Store user message
        self._store_user_message(user_message)
        
        # 2. Retrieve relevant context
        context = self._get_relevant_context(user_message)
        
        # 3. Generate response with context
        ai_response = self._generate_response(user_message, context)
        
        # 4. Store AI response
        self._store_ai_response(ai_response)
        
        return ai_response
    
    def _store_user_message(self, message: str):
        """Store user message in memory"""
        self.memory.add({
            "content": f"User said: {message}",
            "memory_type": "episodic",
            "metadata": {
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "speaker": "user",
                "message_type": "input"
            }
        })
    
    def _store_ai_response(self, response: str, sources: List[str] = None):
        """Store AI response in memory with optional source grounding"""
        memory_item_id = self.memory.add({
            "content": f"I responded: {response}",
            "memory_type": "episodic", 
            "metadata": {
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
                "speaker": "assistant",
                "message_type": "response"
            }
        })
        
        # Ground response to sources if provided
        if sources:
            for source_url in sources:
                self.memory.ground(
                    item_id=memory_item_id,
                    source_url=source_url,
                    validation={
                        "confidence": 0.8,
                        "grounding_type": "ai_response_source",
                        "grounded_at": datetime.now().isoformat()
                    }
                )
    
    def _get_relevant_context(self, message: str, max_context: int = 5) -> List[str]:
        """Retrieve relevant conversation context"""
        # Search for relevant memories
        relevant_memories = self.memory.search(
            query=message,
            memory_type="episodic",
            user_id=self.user_id,
            top_k=max_context
        )
        
        # Extract context strings
        context = []
        for memory in relevant_memories:
            context.append(memory.content)
        
        return context
    
    def _generate_response(self, message: str, context: List[str]) -> str:
        """Generate AI response using OpenAI with memory context"""
        
        # Build context string
        context_str = "\n".join(context) if context else "No previous context."
        
        # Create prompt with context
        prompt = f"""
        Previous conversation context:
        {context_str}
        
        Current user message: {message}
        
        Please respond as a helpful assistant, taking into account the previous context.
        Be conversational and reference previous topics when relevant.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with memory of previous conversations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"

# Usage example
chatbot = MemoryEnhancedChatbot(
    user_id="user123",
    openai_api_key="your-openai-api-key"
)

# Conversation
print(chatbot.chat("Hi, I'm working on a Python project"))
print(chatbot.chat("I'm having trouble with authentication"))
print(chatbot.chat("What was I working on yesterday?"))  # References previous context
```

## Advanced Implementation

### Multi-User Conversational AI with Personality

```python
from smartmemory import SmartMemory
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta

class AdvancedConversationalAI:
    def __init__(self):
        self.memory = SmartMemory()
        self.user_profiles = {}
        
    def chat(self, user_id: str, message: str, session_id: Optional[str] = None) -> Dict:
        """Enhanced chat with user profiling and session management"""
        
        # Get or create user profile
        profile = self._get_user_profile(user_id)
        
        # Store message with rich metadata
        self._store_message(user_id, message, "user", session_id)
        
        # Get personalized context
        context = self._get_personalized_context(user_id, message, session_id)
        
        # Generate response
        response = self._generate_personalized_response(
            user_id, message, context, profile
        )
        
        # Store response
        self._store_message(user_id, response, "assistant", session_id)
        
        # Update user profile
        self._update_user_profile(user_id, message, response)
        
        return {
            "response": response,
            "context_used": len(context),
            "user_profile": profile,
            "session_id": session_id
        }
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """Get or create user personality profile"""
        if user_id not in self.user_profiles:
            # Search for existing profile in memory
            profile_memories = self.memory.search(
                f"user_profile_{user_id}",
                memory_type="semantic",
                user_id=user_id
            )
            
            if profile_memories:
                # Load existing profile
                profile_data = json.loads(profile_memories[0].content)
                self.user_profiles[user_id] = profile_data
            else:
                # Create new profile
                self.user_profiles[user_id] = {
                    "preferences": {},
                    "interests": [],
                    "communication_style": "neutral",
                    "expertise_areas": [],
                    "conversation_count": 0,
                    "first_interaction": datetime.now().isoformat(),
                    "last_interaction": datetime.now().isoformat()
                }
        
        return self.user_profiles[user_id]
    
    def _store_message(self, user_id: str, message: str, speaker: str, session_id: Optional[str]):
        """Store message with rich metadata"""
        self.memory.add({
            "content": f"{speaker}: {message}",
            "memory_type": "episodic",
            "metadata": {
                "user_id": user_id,
                "session_id": session_id or "default",
                "speaker": speaker,
                "timestamp": datetime.now().isoformat(),
                "message_length": len(message),
                "contains_question": "?" in message
            }
        })
    
    def _get_personalized_context(self, user_id: str, message: str, session_id: Optional[str]) -> List[Dict]:
        """Get context personalized for the user"""
        context = []
        
        # Recent session context (high priority)
        if session_id:
            recent_session = self.memory.search(
                query=message,
                memory_type="episodic", 
                user_id=user_id,
                top_k=3
            )
            context.extend([{
                "content": mem.content,
                "relevance": "session",
                "timestamp": mem.metadata.get("timestamp")
            } for mem in recent_session])
        
        # Relevant historical context
        historical = self.memory.search(
            query=message,
            memory_type="episodic",
            user_id=user_id,
            top_k=5
        )
        context.extend([{
            "content": mem.content,
            "relevance": "historical",
            "timestamp": mem.metadata.get("timestamp")
        } for mem in historical])
        
        # User preferences and interests
        profile = self.user_profiles.get(user_id, {})
        for interest in profile.get("interests", []):
            if interest.lower() in message.lower():
                context.append({
                    "content": f"User is interested in {interest}",
                    "relevance": "preference",
                    "timestamp": None
                })
        
        return context[:8]  # Limit context size
    
    def _generate_personalized_response(self, user_id: str, message: str, context: List[Dict], profile: Dict) -> str:
        """Generate response adapted to user's communication style"""
        
        # Build context string with relevance
        context_parts = []
        for ctx in context:
            relevance = ctx["relevance"]
            content = ctx["content"]
            context_parts.append(f"[{relevance}] {content}")
        
        context_str = "\n".join(context_parts)
        
        # Adapt style based on user profile
        style_instructions = self._get_style_instructions(profile)
        
        prompt = f"""
        User Profile:
        - Communication style: {profile.get('communication_style', 'neutral')}
        - Interests: {', '.join(profile.get('interests', []))}
        - Expertise areas: {', '.join(profile.get('expertise_areas', []))}
        - Conversation count: {profile.get('conversation_count', 0)}
        
        Conversation Context:
        {context_str}
        
        Style Instructions: {style_instructions}
        
        Current message: {message}
        
        Respond appropriately considering the user's profile and conversation history.
        """
        
        # Use your preferred LLM here
        return self._call_llm(prompt)
    
    def _get_style_instructions(self, profile: Dict) -> str:
        """Get communication style instructions based on user profile"""
        style = profile.get("communication_style", "neutral")
        
        style_map = {
            "formal": "Use formal language and professional tone",
            "casual": "Use casual, friendly language with contractions",
            "technical": "Use technical terminology and detailed explanations",
            "concise": "Keep responses brief and to the point",
            "enthusiastic": "Use energetic and positive language",
            "neutral": "Use balanced, helpful tone"
        }
        
        return style_map.get(style, style_map["neutral"])
    
    def _update_user_profile(self, user_id: str, user_message: str, ai_response: str):
        """Update user profile based on interaction"""
        profile = self.user_profiles[user_id]
        
        # Update conversation count
        profile["conversation_count"] += 1
        profile["last_interaction"] = datetime.now().isoformat()
        
        # Extract interests from message
        self._extract_interests(user_message, profile)
        
        # Detect communication style
        self._detect_communication_style(user_message, profile)
        
        # Store updated profile
        self.memory.add({
            "content": json.dumps(profile),
            "memory_type": "semantic",
            "metadata": {
                "user_id": user_id,
                "content_type": "user_profile",
                "updated_at": datetime.now().isoformat()
            }
        })
    
    def _extract_interests(self, message: str, profile: Dict):
        """Extract and update user interests"""
        # Simple keyword-based interest detection
        interest_keywords = {
            "programming": ["python", "javascript", "coding", "development"],
            "ai": ["ai", "machine learning", "neural networks", "llm"],
            "sports": ["football", "basketball", "soccer", "tennis"],
            "music": ["music", "guitar", "piano", "singing"],
            "travel": ["travel", "vacation", "trip", "country"]
        }
        
        message_lower = message.lower()
        for interest, keywords in interest_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                if interest not in profile["interests"]:
                    profile["interests"].append(interest)
    
    def _detect_communication_style(self, message: str, profile: Dict):
        """Detect and update communication style"""
        # Simple style detection
        if len(message.split()) > 20:
            profile["communication_style"] = "detailed"
        elif "!" in message or message.isupper():
            profile["communication_style"] = "enthusiastic"
        elif any(word in message.lower() for word in ["please", "thank you", "sir", "madam"]):
            profile["communication_style"] = "formal"
        else:
            profile["communication_style"] = "casual"
    
    def _call_llm(self, prompt: str) -> str:
        """Call your preferred LLM service"""
        # Implement your LLM call here
        # This is a placeholder
        return "This is a placeholder response. Implement your LLM integration here."
    
    def get_conversation_summary(self, user_id: str, days: int = 7) -> Dict:
        """Get conversation summary for a user"""
        # Get recent conversations
        recent_memories = self.memory.search(
            query="",  # Empty query to get all
            memory_type="episodic",
            user_id=user_id,
            top_k=50
        )
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_conversations = []
        
        for memory in recent_memories:
            timestamp_str = memory.metadata.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp > cutoff_date:
                    recent_conversations.append(memory)
        
        # Analyze conversations
        total_messages = len(recent_conversations)
        user_messages = len([m for m in recent_conversations if "User said:" in m.content])
        ai_messages = len([m for m in recent_conversations if "I responded:" in m.content])
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "days_analyzed": days,
            "average_daily_messages": total_messages / days if days > 0 else 0,
            "recent_topics": self._extract_recent_topics(recent_conversations)
        }
    
    def _extract_recent_topics(self, conversations: List) -> List[str]:
        """Extract main topics from recent conversations"""
        # Simple topic extraction - could be enhanced with NLP
        all_text = " ".join([conv.content for conv in conversations])
        
        # Basic keyword frequency
        words = all_text.lower().split()
        word_freq = {}
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "said", "responded", "user", "i"}
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top topics
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:5]]

# Usage example
ai = AdvancedConversationalAI()

# Multi-turn conversation
user_id = "alice123"
session_id = "session_001"

response1 = ai.chat(user_id, "Hi! I'm learning Python programming", session_id)
print(f"AI: {response1['response']}")

response2 = ai.chat(user_id, "I'm having trouble with loops", session_id)
print(f"AI: {response2['response']}")

response3 = ai.chat(user_id, "What was I learning about earlier?", session_id)
print(f"AI: {response3['response']}")

# Get conversation summary
summary = ai.get_conversation_summary(user_id, days=7)
print(f"Conversation summary: {summary}")
```

## Integration with Popular Frameworks

### LangChain Integration

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from smartmemory import SmartMemory

class SmartMemoryLangChain:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.smart_memory = SmartMemory()
        self.conversation_memory = ConversationBufferMemory()
        self.llm = OpenAI(temperature=0.7)
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.conversation_memory
        )
    
    def chat(self, message: str) -> str:
        # Get relevant long-term context
        context = self.smart_memory.search(
            query=message,
            user_id=self.user_id,
            top_k=3
        )
        
        # Add context to conversation
        if context:
            context_str = "\n".join([mem.content for mem in context])
            enhanced_message = f"Context: {context_str}\n\nUser: {message}"
        else:
            enhanced_message = message
        
        # Get response from LangChain
        response = self.chain.predict(input=enhanced_message)
        
        # Store in SmartMemory for long-term retention
        self.smart_memory.add({
            "content": f"User: {message}",
            "memory_type": "episodic",
            "metadata": {"user_id": self.user_id}
        })
        
        self.smart_memory.add({
            "content": f"Assistant: {response}",
            "memory_type": "episodic", 
            "metadata": {"user_id": self.user_id}
        })
        
        return response
```

### Streamlit Chat Interface

```python
import streamlit as st
from smartmemory import SmartMemory

def create_chat_interface():
    st.title("Memory-Enhanced Chatbot")
    
    # Initialize session state
    if "memory" not in st.session_state:
        st.session_state.memory = SmartMemory()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to talk about?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get relevant context
        context = st.session_state.memory.search(prompt, top_k=3)
        
        # Generate response (simplified)
        if context:
            response = f"Based on our previous conversations about {', '.join([c.content[:50] for c in context])}, I think..."
        else:
            response = "I don't have previous context about this topic, but I can help you with..."
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Store conversation in memory
        st.session_state.memory.add(f"User: {prompt}")
        st.session_state.memory.add(f"Assistant: {response}")

if __name__ == "__main__":
    create_chat_interface()
```

## Best Practices

### Memory Management

1. **User Isolation**: Always use user_id to separate conversations
2. **Session Management**: Use session_id for conversation context
3. **Memory Types**: Use episodic for conversations, semantic for facts
4. **Context Limits**: Limit context size to avoid overwhelming the LLM

### Performance Optimization

1. **Background Processing**: Use `ingest()` for real-time conversations
2. **Caching**: Cache user profiles and frequent queries
3. **Batch Operations**: Process multiple messages together when possible

### Privacy and Security

1. **Data Encryption**: Encrypt sensitive conversation data
2. **User Consent**: Get explicit consent for memory storage
3. **Data Retention**: Implement retention policies
4. **Access Control**: Restrict access to user-specific memories

## Next Steps

- [Learning Assistant Example](learning-assistant) - Educational AI with memory
- [Knowledge Graph Example](knowledge-graph) - Build knowledge networks
- [Advanced Features](../guides/advanced-features) - Explore advanced capabilities
- [MCP Integration](../guides/mcp-integration) - Connect with LLM agents
