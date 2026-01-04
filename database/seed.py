from .connection import get_connection


def seed_database() -> None:
    """Seed the database with sample data."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Check if already seeded
        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] > 0:
            return

        # Seed customers
        customers = [
            ("Alice Johnson", "alice@example.com", "555-0101"),
            ("Bob Smith", "bob@example.com", "555-0102"),
            ("Carol White", "carol@example.com", "555-0103"),
            ("David Brown", "david@example.com", "555-0104"),
            ("Eva Martinez", "eva@example.com", "555-0105"),
        ]
        cursor.executemany(
            "INSERT INTO customers (name, email, phone) VALUES (?, ?, ?)", customers
        )

        # Seed products
        products = [
            (
                "Wireless Headphones",
                "Premium noise-canceling headphones",
                149.99,
                50,
                "Electronics",
            ),
            ("USB-C Hub", "7-in-1 USB-C adapter", 49.99, 120, "Electronics"),
            (
                "Mechanical Keyboard",
                "RGB mechanical keyboard with Cherry MX switches",
                129.99,
                35,
                "Electronics",
            ),
            (
                "Laptop Stand",
                "Adjustable aluminum laptop stand",
                39.99,
                80,
                "Accessories",
            ),
            ("Webcam HD", "1080p webcam with microphone", 79.99, 45, "Electronics"),
            ("Mouse Pad XL", "Extended gaming mouse pad", 24.99, 200, "Accessories"),
            ("Monitor Light", "LED monitor light bar", 59.99, 60, "Accessories"),
            (
                "Cable Organizer",
                "Desktop cable management kit",
                19.99,
                150,
                "Accessories",
            ),
        ]
        cursor.executemany(
            "INSERT INTO products (name, description, price, stock_quantity, category) VALUES (?, ?, ?, ?, ?)",
            products,
        )

        # Seed orders
        orders = [
            (1, "delivered", 199.98, "123 Main St, City A", "TRK001234"),
            (1, "shipped", 129.99, "123 Main St, City A", "TRK001235"),
            (2, "processing", 49.99, "456 Oak Ave, City B", None),
            (3, "pending", 169.98, "789 Pine Rd, City C", None),
            (4, "delivered", 79.99, "321 Elm St, City D", "TRK001236"),
            (5, "cancelled", 149.99, "654 Maple Dr, City E", None),
        ]
        cursor.executemany(
            "INSERT INTO orders (customer_id, status, total_amount, shipping_address, tracking_number) VALUES (?, ?, ?, ?, ?)",
            orders,
        )

        # Seed order items
        order_items = [
            (1, 1, 1, 149.99),  # Order 1: Wireless Headphones
            (1, 2, 1, 49.99),  # Order 1: USB-C Hub
            (2, 3, 1, 129.99),  # Order 2: Mechanical Keyboard
            (3, 2, 1, 49.99),  # Order 3: USB-C Hub
            (4, 1, 1, 149.99),  # Order 4: Wireless Headphones
            (4, 8, 1, 19.99),  # Order 4: Cable Organizer
            (5, 5, 1, 79.99),  # Order 5: Webcam HD
            (6, 1, 1, 149.99),  # Order 6: Wireless Headphones (cancelled)
        ]
        cursor.executemany(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
            order_items,
        )

        conn.commit()
