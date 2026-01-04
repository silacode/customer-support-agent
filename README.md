# Customer Support Agent

A CLI-based customer support agent powered by OpenAI with SQL database and RAG tools.

## Features

- **Order Management**: Query customer orders, order status, and tracking information
- **Product Lookup**: Check product details, pricing, and stock levels
- **Policy Search**: RAG-powered search across company policies (returns, shipping, warranty)
- **Conversation History**: Maintains context across multiple exchanges

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Interface                       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    OpenAI Agent                          │
│                 (Function Calling)                       │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
┌──────────▼──────────┐      ┌───────────▼────────────────┐
│   SQL Tool          │      │   RAG Tool                 │
│   (SQLite)          │      │   (ChromaDB + Embeddings)  │
└─────────────────────┘      └────────────────────────────┘
```

## Setup

1. **Clone and install dependencies**:
   ```bash
   cd customer_support_agent
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the agent**:
   ```bash
   uv run main.py
   ```

## Usage

```
╭──────────────────── Welcome ────────────────────╮
│         Customer Support Agent                  │
│                                                 │
│ Ask me about orders, products, stock levels,    │
│ or company policies.                            │
│                                                 │
│ Commands:                                       │
│   quit or exit - Exit the agent                 │
│   clear - Clear conversation history            │
│   help - Show this message                      │
╰─────────────────────────────────────────────────╯

You: What's the status of order 2?
Agent: Order #2 is currently **shipped** and on its way to the customer...

You: What's your return policy?
Agent: Our return policy allows returns within 30 days of delivery...
```

## Project Structure

```
customer_support_agent/
├── main.py              # CLI entry point
├── agent/
│   ├── core.py          # OpenAI agent with tool calling
│   └── tools.py         # Tool definitions and execution
├── database/
│   ├── connection.py    # SQLite connection manager
│   ├── models.py        # Database schema
│   └── seed.py          # Sample data seeding
├── rag/
│   ├── embeddings.py    # OpenAI embeddings via httpx
│   ├── loader.py        # Policy document loader
│   └── vectorstore.py   # ChromaDB operations
├── policies/            # Policy markdown files
│   ├── returns.md
│   ├── shipping.md
│   └── warranty.md
└── data/                # SQLite & ChromaDB storage (gitignored)
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-5-mini` |
| `DATABASE_PATH` | SQLite database path | `data/support.db` |
| `CHROMA_PATH` | ChromaDB storage path | `data/chroma` |

## Sample Data

The agent comes pre-seeded with:
- 5 sample customers
- 8 products (electronics and accessories)
- 6 orders with various statuses

## License

MIT

