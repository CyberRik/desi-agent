from desi_agent.eval.guard import grounded


def test_invented_card_is_blocked_real_one_passes():
    assert not grounded("block_card", {"card_last4": "1234", "reason": "lost"}, "I lost my card, block it", "")
    assert grounded("block_card", {"card_last4": "4521", "reason": "lost"}, "card ending 4521 is lost", "")


def test_dispute_id_may_come_from_a_statement():
    st = '{"transactions": [{"transaction_id": "TXN806078"}]}'
    assert grounded("raise_dispute", {"transaction_id": "TXN806078", "reason": "duplicate"}, "dispute bigbasket", st)
    assert not grounded("raise_dispute", {"transaction_id": "TXN123456", "reason": "duplicate"}, "dispute it", st)


def test_transfer_needs_upi_and_amount_from_customer():
    assert grounded("transfer_money", {"to_upi_id": "ramesh@okaxis", "amount": 1500}, "send 1,500 to ramesh@okaxis", "")
    assert grounded("transfer_money", {"to_upi_id": "a@ybl", "amount": 500000}, "a@ybl ko 5 lakh bhejo", "")
    assert not grounded("transfer_money", {"to_upi_id": "a@ybl", "amount": 2000}, "send money to a@ybl", "")


def test_read_only_tools_never_blocked():
    assert grounded("get_account_balance", {"account_type": "savings"}, "", "")
