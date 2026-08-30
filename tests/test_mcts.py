import time

import chess
import pytest

import agent
import mcts


def make_searcher():
    return mcts.Searcher(agent.evaluate, batch_size=8)


def _stub_evaluate(value=0.0):
    """Deterministic, network-free evaluate(): uniform priors over legal
    moves, a constant value. Used for white-box tests of the search's
    bookkeeping (sign convention, virtual loss, visit accounting) where the
    real (random-weight) network's opinions would only add noise."""

    def fn(boards):
        priors, values = [], []
        for b in boards:
            moves = list(b.legal_moves)
            if moves:
                p = 1.0 / len(moves)
                priors.append({m: p for m in moves})
            else:
                priors.append({})
            values.append(value)
        return priors, values

    return fn


def _walk(node, out):
    out.append(node)
    if node.children:
        for child in node.children.values():
            _walk(child, out)


def test_finds_mate_in_one():
    # Ra8# is the only mate; terminal backup must dominate any net prior.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    move = make_searcher().search(board, time.perf_counter() + 2.0)
    assert move == chess.Move.from_uci("a1a8")


def test_respects_deadline():
    board = chess.Board()
    start = time.perf_counter()
    make_searcher().search(board, start + 0.5)
    assert time.perf_counter() - start < 1.0


def test_returns_legal_in_forced_positions():
    # Two legal moves (Kh8-g8/g7/h7 all attacked or blocked but for these
    # two) - this genuinely exercises _run_batch, unlike a single-legal-move
    # position which returns via the root shortcut without searching at all.
    board = chess.Board("8/7p/8/8/8/2k5/8/K7 w - - 0 1")
    assert len(list(board.legal_moves)) >= 2
    move = make_searcher().search(board, time.perf_counter() + 0.3)
    assert move in board.legal_moves


def test_single_legal_move_shortcut():
    # Only one legal move: king must take. Exercises the root shortcut path
    # (no batches run) - kept as its own test, separate from the
    # >=2-legal-move test above.
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    move = make_searcher().search(board, time.perf_counter() + 0.3)
    assert move == chess.Move.from_uci("h1g2")


def test_board_unmodified_after_search():
    # The searcher pushes/pops moves on the caller's board; it must leave
    # it exactly as it found it (same FEN, same move stack).
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    fen_before = board.fen()
    stack_before = list(board.move_stack)
    make_searcher().search(board, time.perf_counter() + 0.3)
    assert board.fen() == fen_before
    assert list(board.move_stack) == stack_before


def test_terminal_root_raises_immediately():
    # Already-checkmated position (Black to move, no legal moves). There is
    # no legal move to return; the searcher must fail fast rather than burn
    # the whole deadline running pointless batches first.
    board = chess.Board("R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1")
    assert list(board.legal_moves) == []
    start = time.perf_counter()
    with pytest.raises(ValueError):
        make_searcher().search(board, start + 5.0)
    assert time.perf_counter() - start < 0.5


def test_backup_alternates_sign():
    # Direct, synthetic test of _backup's sign convention: it must alternate
    # at every ply on the way back to the root, not apply a single sign to
    # the whole path. Nodes are seeded as if virtual loss had already been
    # applied once at selection time (value_sum = -_VIRTUAL_LOSS), matching
    # real usage; _backup must undo exactly that and add the alternating
    # value on top.
    searcher = mcts.Searcher(_stub_evaluate())

    for length, expected in ((2, [0.7, -0.7]), (3, [-0.7, 0.7, -0.7])):
        nodes = [mcts._Node(1.0) for _ in range(length)]
        for node in nodes:
            node.visits = 1
            node.value_sum = -mcts._VIRTUAL_LOSS
        path = [(None, node) for node in nodes]
        searcher._backup(path, 0.7)
        assert [node.value_sum for node in nodes] == expected


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_virtual_loss_cancels_exactly(batch_size):
    # From a quiet position with a constant-0 stub value, every completed
    # traversal's virtual loss is applied once at selection and undone
    # exactly once at backup (terminal or not). Over many batches, every
    # node in the tree must end up with value_sum == 0.0 EXACTLY - any
    # residue means virtual loss isn't being fully undone somewhere.
    board = chess.Board()
    searcher = mcts.Searcher(_stub_evaluate(0.0), batch_size=batch_size)
    root = mcts._Node(1.0)
    priors, _ = searcher.evaluate([board])
    root.children = {m: mcts._Node(p) for m, p in priors[0].items()}

    for _ in range(5):
        searcher._run_batch(board, root)

    nodes = []
    _walk(root, nodes)
    assert len(nodes) > 1  # sanity: the tree actually grew
    for node in nodes:
        if node is root:
            continue
        assert node.value_sum == 0.0


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_visit_accounting_exact(batch_size):
    # sum(child.visits) over root's children must equal the number of
    # traversals that actually completed (leaves collected + terminal
    # backups), since every traversal passes through exactly one root
    # child and a real bug in visit bookkeeping (double-increment, missing
    # decrement in the abandon path, etc.) would throw this off.
    board = chess.Board()
    searcher = mcts.Searcher(_stub_evaluate(0.0), batch_size=batch_size)
    root = mcts._Node(1.0)
    priors, _ = searcher.evaluate([board])
    root.children = {m: mcts._Node(p) for m, p in priors[0].items()}

    completed = 0
    for _ in range(5):
        leaves, terminal_backups = searcher._run_batch(board, root)
        completed += len(leaves) + terminal_backups

    assert sum(c.visits for c in root.children.values()) == completed


def test_deep_sign_convention():
    # Kh1 walks into ...Qa1# (a depth-2 forced mate against White). With a
    # value-blind (constant 0) stub, only the alternating terminal backup
    # can teach the search that g1h1 is bad - a same-sign or non-alternating
    # bug would leave g1h1's Q near 0 like every other move, or even
    # positive. This is the depth-2 canary the mate-in-one test can't cover
    # (mate-in-one only exercises a single ply of backup).
    board = chess.Board("q6k/8/8/8/8/8/5PPP/6K1 w - - 0 1")
    searcher = mcts.Searcher(_stub_evaluate(0.0), batch_size=64)
    root = mcts._Node(1.0)
    priors, _ = searcher.evaluate([board])
    root.children = {m: mcts._Node(p) for m, p in priors[0].items()}

    kh1 = chess.Move.from_uci("g1h1")
    for _ in range(40):
        searcher._run_batch(board, root)

    assert root.children[kh1].visits > 0
    q_kh1 = root.children[kh1].value_sum / root.children[kh1].visits
    assert q_kh1 < -0.1, f"g1h1 should look clearly bad for White, got Q={q_kh1}"

    most_visited = max(root.children, key=lambda m: root.children[m].visits)
    assert most_visited != kh1


def test_puct_uses_sqrt_parent_visits():
    # Direct, synthetic test of the PUCT formula in _select_child: an
    # unvisited child must eventually get selected no matter how tiny its
    # prior is, as long as the parent's total visits keeps growing - because
    # AlphaZero's exploration term is c_puct * prior * sqrt(N_parent), which
    # is unbounded in N_parent. If the sqrt(N_parent) factor is dropped, an
    # unvisited child's score is stuck at a fixed c_puct * prior forever and
    # can never overcome a sibling with a steady, decent Q - "rare" would
    # never be explored no matter how long the search runs. Note: within a
    # single _select_child call, sqrt(N_parent) is a common factor across
    # all of that node's children, so it never changes *which* sibling wins
    # a single comparison - only across many calls, as N_parent grows, does
    # its absence change behavior. That's why this needs to run the
    # selection loop out far enough to observe the crossover, rather than
    # checking one call's scores.
    searcher = mcts.Searcher(_stub_evaluate())
    node = mcts._Node(1.0)
    favored = mcts._Node(0.9)
    favored.visits = 1
    favored.value_sum = 0.5  # Q = 0.5, held steady by the backup below
    rare = mcts._Node(0.001)  # tiny prior, stays unvisited until sqrt(N) rescues it
    node.children = {"favored": favored, "rare": rare}

    picked_rare = False
    for _ in range(200_000):
        child = searcher._select_child(node, [])
        if child is rare:
            picked_rare = True
            break
        # Settle the virtual loss straight back to Q=0.5 so "favored" stays
        # a steady, decent sibling rather than decaying away from
        # unaccompanied virtual loss - isolating the sqrt(N) dependence.
        searcher._backup([(None, favored)], -0.5)

    assert picked_rare, (
        "an unvisited, tiny-prior child was never explored even after "
        "200000 selections - PUCT's exploration term isn't growing with "
        "the parent's visit count"
    )


def test_duplicate_leaf_abandon_path():
    # White has exactly two legal moves (a1a2, a1b1), neither of which ends
    # the game. With batch_size=64 the first two selections claim the two
    # distinct unexpanded leaves; every further selection in the same batch
    # must re-hit one of them, triggering the duplicate-leaf abandon path
    # (_undo_virtual) instead of double-counting a visit. We spy on
    # _undo_virtual to prove the path actually ran, and check the same
    # exact-visit-accounting invariant as above to prove it unwound cleanly.
    board = chess.Board("8/7p/8/8/8/2k5/8/K7 w - - 0 1")
    assert len(list(board.legal_moves)) == 2
    searcher = mcts.Searcher(_stub_evaluate(0.0), batch_size=64)
    root = mcts._Node(1.0)
    priors, _ = searcher.evaluate([board])
    root.children = {m: mcts._Node(p) for m, p in priors[0].items()}

    undo_calls = []
    original_undo = searcher._undo_virtual

    def spy(path):
        undo_calls.append(path)
        return original_undo(path)

    searcher._undo_virtual = spy

    fen_before = board.fen()
    leaves, terminal_backups = searcher._run_batch(board, root)

    assert len(undo_calls) >= 1, "abandon path never fired - test setup is wrong"
    assert board.fen() == fen_before
    completed = len(leaves) + terminal_backups
    assert sum(c.visits for c in root.children.values()) == completed
    # With only 2 legal moves and no ties broken by prior differences, both
    # should be visited exactly once before the third selection collides.
    assert completed == 2
    assert all(c.visits == 1 for c in root.children.values())
