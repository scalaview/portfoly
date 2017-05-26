import sympy as sy



x = sy.Symbol('x')
y = sy.Symbol('y')
type(x)
<class 'sympy.core.symbol.Symbol'>
sy.sqrt(x)
sqrt(x)
3 + sy.sqrt(x) - 4**2
sqrt(x) - 13
f = x ** 2 + 3 + 0.5*x **2 + 3/2
>>>
sy.simplify(f)
1.5*x**2 + 4.5
sy.init_printing(pretty_print=False, use_unicode=False)
print(sy.pretty(f))
     2
1.5*x  + 4.5
print(sy.pretty(sy.sqrt(x) + 0.5))
  ___
\/ x  + 0.5
# 解方程
sy.solve(x ** 2 - 1)
[-1, 1]
sy.solve(x ** 3 + 0.5 * x ** 2 - 1)
[0.858094329496553, -0.679047164748276 - 0.839206763026694*I, -0.679047164748276 + 0.839206763026694*I]
sy.solve(x ** 2 + y ** 2)
[{x: -I*y}, {x: I*y}]

# 积分
a, b = sy.symbols('a b')
print(sy.pretty(sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))))
  b
  /
 |
 |  (0.5*x + sin(x)) dx
 |
/
a
# 积分
# 求出反导
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
print(sy.pretty(int_func))
      2
0.25*x  - cos(x)
Fb = int_func.subs(x, 0.95).evalf()
Fa = int_func.subs(x, 0.5).evalf()
Fb = int_func.subs(x, 9.5).evalf()
Fb - Fa
24.3747547180867

# 求出双界的反到公式
int_func_limts = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
print(sy.pretty(int_func_limts))
        2         2
- 0.25*a  + 0.25*b  + cos(a) - cos(b)
int_func_limts.subs({a: 0.5, b: 9.5}).evalf()
24.3747547180868
# 直接计算积分
sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5))
24.3747547180867



f = (sy.sin(x) + 0.05 * x ** 2 + sy.sin(y) + 0.05 * y ** 2)
# 对x求偏导
del_x = sy.diff(f, x)
del_x
0.1*x + cos(x)

# 对y求偏导
del_y = sy.diff(f, y)
del_y
0.1*y + cos(y)

'''  求解方程，得不到解，因此需要使用近似解
>>> sy.solve(del_x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/zbin/anaconda/lib/python3.6/site-packages/sympy/solvers/solvers.py", line 1053, in solve
    solution = _solve(f[0], *symbols, **flags)
  File "/Users/zbin/anaconda/lib/python3.6/site-packages/sympy/solvers/solvers.py", line 1619, in _solve
    raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))
NotImplementedError: multiple generators [x, cos(x)]
No algorithms are implemented to solve equation x/10 + cos(x)
'''
# 求得近似解，后面的参数表示从某个数字（x）开始得到第一近似解
>>> xo = sy.nsolve(del_x, -1.5)
>>> xo
-1.42755177876459

>>> yo = sy.nsolve(del_y, -1.5)
>>> yo
-1.42755177876459
