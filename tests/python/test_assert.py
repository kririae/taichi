import pytest
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_minimal():
    @ti.kernel
    def func():
        assert 0

    @ti.kernel
    def func2():
        assert False

    with pytest.raises(AssertionError):
        func()
    with pytest.raises(AssertionError):
        func2()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_basic():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20

    with pytest.raises(AssertionError):
        func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20, "Foo bar"

    with pytest.raises(AssertionError, match="Foo bar"):
        func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted():
    x = ti.field(dtype=int, shape=16)
    x[10] = 42

    @ti.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, "x[%d] expect=%d got=%d" % (i, 0, x[i])

    @ti.kernel
    def assert_float():
        y = 0.5
        assert y < 0, "y = %f" % y

    with pytest.raises(AssertionError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(AssertionError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted_fstring():
    x = ti.field(dtype=int, shape=16)
    x[10] = 42

    @ti.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, f"x[{i}] expect={0} got={x[i]}"

    @ti.kernel
    def assert_float():
        y = 0.5
        assert y < 0, f"y = {y}"

    with pytest.raises(AssertionError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(AssertionError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_ok():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x <= 20

    func()


@test_utils.test(
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_assert_with_check_oob():
    @ti.kernel
    def func():
        n = 15
        assert n >= 0

    func()


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False, remove_assertion=True)
def test_assert_without_side_effect_debug():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return False

    @ti.kernel
    def func():
        assert side_effect()

    func()
    assert x[None] == 0


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False, remove_assertion=False)
def test_assert_with_side_effect_debug():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return True

    @ti.kernel
    def func():
        assert side_effect()

    func()
    assert x[None] == 1


@test_utils.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_debug_default():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return True

    @ti.kernel
    def func():
        assert side_effect()

    func()
    assert x[None] == 1


@test_utils.test(require=ti.extension.assertion, debug=False, gdb_trigger=False, remove_assertion=True)
def test_assert_without_side_effect_release():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return False

    @ti.kernel
    def func():
        assert side_effect()

    func()
    assert x[None] == 0


@test_utils.test(require=ti.extension.assertion, debug=False, gdb_trigger=False, remove_assertion=False)
def test_assert_with_side_effect_release():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return True

    @ti.kernel
    def func():
        assert side_effect()

    func()
    assert x[None] == 1


@test_utils.test(require=ti.extension.assertion, debug=False, gdb_trigger=False)
def test_assert_release_default():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return False

    @ti.kernel
    def func():
        assert side_effect()

    func()
    assert x[None] == 0


@test_utils.test(require=ti.extension.assertion, debug=False, gdb_trigger=False, remove_assertion=False)
def test_assert_release_trigger():
    x = ti.field(dtype=int, shape=())

    @ti.func
    def side_effect():
        x[None] = 1
        return False

    @ti.kernel
    def func():
        assert side_effect()

    with pytest.raises(AssertionError):
        func()
    assert x[None] == 1


@test_utils.test(
    require=ti.extension.assertion, debug=False, gdb_trigger=False, remove_assertion=True, check_out_of_bound=True
)
def test_assert_release_trigger_out_of_bounds():
    idx = ti.field(dtype=int, shape=())
    x = ti.field(dtype=int, shape=(1,))

    @ti.kernel
    def func():
        x[idx[None]] = 1

    idx[None] = 0
    func()

    with pytest.raises(AssertionError):
        idx[None] = 1
        func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_message():
    x = 3

    @ti.kernel
    def func():
        ti.static_assert(x == 4, "Oh, no!")

    with pytest.raises(ti.TaichiCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_vector_n_ok():
    x = ti.Vector.field(4, ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.n == 4)

    func()


@test_utils.test(arch=get_host_arch_list())
def test_static_assert_data_type_ok():
    x = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        ti.static_assert(x.dtype == ti.f32)

    func()


@test_utils.test()
def test_static_assert_nonstatic_condition():
    @ti.kernel
    def foo():
        value = False
        ti.static_assert(value, "Oh, no!")

    with pytest.raises(ti.TaichiTypeError, match="Static assert with non-static condition"):
        foo()
