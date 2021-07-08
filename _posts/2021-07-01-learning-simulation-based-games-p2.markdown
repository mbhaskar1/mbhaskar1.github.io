---
layout: post
title:  "Paper Walkthrough: Learning Equilibria in Simulation Based Games - Part 2"
date:   2021-07-08
categories: [paper-walkthrough]
excerpt: This is part 2 of my Paper Walkthrough of the paper "Learning Equilibria in Simulation Based Games" by Enrique Areyan Viqueria, Cyrus Cousins, Eli Upfal, and Amy Greenwald [1]. If you have not read part 1, please go back and read it first. I will begin this part by doing a small recap of what has been covered so far...
---

This is part 2 of my Paper Walkthrough of the paper ["Learning Equilibria in Simulation Based Games"](http://arxiv.org/abs/1905.13379) by Enrique Areyan Viqueria, Cyrus Cousins, Eli Upfal, and Amy Greenwald [1]. If you have not read [part 1](https://mbhaskar1.github.io/paper-walkthrough/2021/06/17/learning-simulation-based-games-p1.html), please go back and read it first. I will begin this part by doing a small recap of what has been covered so far.

In part 1 of this Paper Walkthrough, we primarily laid down the groundwork for the problem that is going to be analyzed. We introduced several formal game representations, including **expected normal-form games** and **empirical normal-form games**, the latter being what we use to represent simulation-based games. We formulated a metric $$d_\infty$$ for measuring the closeness between two normal-form games, in this case an empirical normal-form game and the expected normal-form game it is approximating, and we argued for the appropriateness of the metric given our use case. Finally, in the spirit of the Probably Approximately Correct Learning Framework, we decided to direct our analysis towards searching for guarantees about closeness that hold with high probability, but not necessarily with certainty. More precisely, given a sample $$X\sim \mathscr{D}^m$$ used to create an empirical normal-form game $$\Gamma_X = \langle P, S, \hat{u}\rangle$$ approximating an expected normal-form game $$\Gamma_\mathscr{D} = \langle P, S, u\rangle$$, and an error-rate $$\delta\in (0, 1)$$, we are searching for $$\epsilon>0$$ that result in strong guarantees of the form

$$
\begin{equation*}
    \textrm{P}(d_\infty(\Gamma_X, \Gamma_\mathscr{D})\leq \epsilon)\geq 1 - \delta.
\end{equation*}
$$

We ended Part 1 having formulated the above problem. In this part, we will explore the methods used by the paper to tackle it.

# Approach 1 - Hoeffding's Inequality

The first approach used to form guarantees about $$d_\infty(\Gamma_X, \Gamma_\mathscr{D})$$ is based on a theorem known as **Hoeffding's Inequality**. Before getting into this theorem, we're going to use the definition of $$d_\infty$$ to rewrite our condition on $$\epsilon$$ as follows:

$$
\begin{equation*}
    \textrm{P}(\forall p, s\in P\times S:\,\, \vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert\leq \epsilon)\geq 1 - \delta.\quad\textbf{(1)}
\end{equation*}
$$

The Bonferroni inequality, also known as the **union bound**, states that for a countable set of events $$E_1, E_2, \dots, E_n$$, we have $$\textrm{P}(\bigcup_{i=1}^n E_i)\leq \sum_{i=1}^n P(E_i)$$. The inequality can be proven easily using induction, and the fact that $$\textrm{P}(A\cup B) = \textrm{P}(A) + \textrm{P}(B) - \textrm{P}(A\cap B)$$. Using basic probability rules, the union bound can be used to derive the following inequality:

$$
\begin{equation*}
    \textrm{P}\left(\bigcap_{i=1}^n E_i\right) = 1 - \textrm{P}\left(\bigcup_{i=1}^n \neg E_i\right) \geq 1 - \sum_{i=1}^n \textrm{P}(\neg E_i).
\end{equation*}
$$

Applying this inequality to the left side of our condition from above, we get

$$
\begin{align*}
    \textrm{P}(\forall p, s\in P\times S:\,\, \vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert\leq \epsilon) &\geq 1 - \sum_{p, s\in P\times S} \textrm{P}(\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert > \epsilon)\\
    &\geq 1 - \vert P \times S\vert \max_{p, s\in P\times S}\textrm{P}(\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert > \epsilon)
\end{align*}
$$

As a result of this, in order to search for small $$\epsilon$$ satisfying condition $$\textbf{(1)}$$, we can search for $$\epsilon$$ satisfying the stronger condition that $$1 - \vert P \times S\vert \max_{p, s\in P\times S}\textrm{P}(\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert > \epsilon)\geq 1 - \delta$$, or equivalently, that for all $$p, s\in P\times S$$, it is the case that

$$
\begin{equation*}
    P(\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert > \epsilon) \leq \frac{\delta}{\vert P\times S\vert}.
\end{equation*}
$$

We will use Hoeffding's Inequality to find $$\epsilon$$ satisfying this new condition. In order to build up to Hoeffding's Inequality, we will first prove some preliminary theorems. The following explanation of Hoeffding's Inequality is based directly on lecture notes on Hoeffding's Inequality by Dr. Clayton Scott from University of Michigan [3].

The first theorem we will look at is **Markov's Inequality**. This theorem is taught in introductory statistics courses, but since the proof is so short, I will include it below.

***

**Markov's Inequality:** If $$U$$ is a non-negative random variable on $$\mathbb{R}$$, then for all $$t>0$$,

$$
\begin{equation*}
    \textrm{P}(U\geq t)\leq \frac{\mathbb{E}[U]}{t}.
\end{equation*}
$$

*Proof:* Let $$U$$ be a non-negative random variable on $$\mathbb{R}$$. Since $$U$$ is non-negative, we have

$$
\begin{equation*}
    \textrm{P}(U\geq t) = \mathbb{E}\left[\textbf{1}_{\{U\geq t\}}\right]\leq \mathbb{E}\left[\frac{U}{t}\textbf{1}_{\{U\geq t\}}\right] = \frac{1}{t}\mathbb{E}\left[U\textbf{1}_{\{U\geq t\}}\right]\leq \frac{1}{t}\mathbb{E}[U]. 
\end{equation*}
$$

Since $$U$$ was arbitrary, we are done.

***

Using Markov's Inequality, we can derive a corollary known as **Chernoff's Bounding Method**.

***

**Chernoff's Bounding Method:** Let $$Z$$ be a random variable on $$\mathbb{R}$$. Then for all $$t> 0$$,

$$
\begin{equation*}
    \textrm{P}(Z\geq t) \leq \inf_{s > 0}\frac{\mathbb{E}\left[e^{sZ}\right]}{e^{st}}.
\end{equation*}
$$

*Proof:* For any $$s > 0$$, we can use Markov's inequality to get

$$
\begin{equation*}
    \textrm{P}(Z\geq t) = \textrm{P}(sZ\geq st) = \textrm{P}(e^{sZ}\geq e^{st})\leq \frac{\mathbb{E}\left[e^{sZ}\right]}{e^{st}}.
\end{equation*}
$$

Since $$\textrm{P}(Z\geq t)\leq \frac{\mathbb{E}\left[e^{sZ}\right]}{e^{st}}$$ for all $$s> 0$$, it must be the case that

$$
\begin{equation*}
    \textrm{P}(Z\geq t) \leq \inf_{s> 0}\frac{\mathbb{E}\left[e^{sZ}\right]}{e^{st}}.
\end{equation*}
$$

***

Using Chernoff's Bounding Method, along with Taylor's Theorem with the Lagrange form of the remainder from calculus, we will now state and prove a lemma known as **Hoeffding's Lemma**, which we will then use to finally prove Hoeffding's Inequality.

***

**Hoeffding's Lemma:** Let $$V$$ be a random variable on $$\mathbb{R}$$ with $$\mathbb{E}[V]=0$$, and suppose that $$a\leq V\leq b$$ with probability one, where $$a < b$$. Then for all $$s > 0$$

$$
\begin{equation*}
    \mathbb{E}[e^{sV}]\leq e^{s^2(b-a)^2/8}
\end{equation*}
$$

*Proof:* Let $$s>0$$. Since the second-derivative of the function $$x\mapsto e^{sx}$$ is always positive, we know that it is a convex function. Recall that any line segment connecting points on the graph of a real-valued convex function will lie at or above the graph between the two points. This gives us that for all $$x\in [a, b]$$ and $$p\in [0, 1]$$, it is the case that

$$
\begin{equation*}
    e^{sx} \leq pe^{sa} + (1 - p)e^{sb}.
\end{equation*}
$$

Using $$p=\frac{V-a}{b-a}$$ and $$x = V$$, this gives us

$$
\begin{equation*}
    e^{sV} \leq \frac{V - a}{b - a}e^{sa} + \frac{b - V}{b - a}e^{sb}.
\end{equation*}
$$

Taking the expectation of both sides of the above inequality, we get

$$
\begin{equation*}
    \mathbb{E}[e^{sV}] \leq \mathbb{E}\left[\frac{V - a}{b - a}e^{sa} + \frac{b - V}{b - a}e^{sb}\right].
\end{equation*}
$$

Notice that the function $$V\mapsto \frac{V - a}{b - a}e^{sa} + \frac{b - V}{b - a}e^{sb}$$ is a straight line and is therefore convex, meeting the conditions for Jensen's Inquality. Applying Jensen's Inequality and the fact that $$\mathbb{E}[V] = 0$$, we get

$$
\begin{align*}
    \mathbb{E}[e^{sV}] &\leq \frac{\mathbb{E}[V] - a}{b - a}e^{sa} + \frac{b - \mathbb{E}[V]}{b - a}e^{sb}\\
    &= \frac{b}{b - a}e^{sb} - \frac{a}{b - a}e^{sa}.\quad\textbf{(2)}
\end{align*}
$$

Now let $$p:= \frac{b}{b - a}$$ and, with the substitution $$u=(b-a)s$$, consider the function:

$$
\begin{align*}
    \varphi(u) &= \log\left(\frac{b}{b - a}e^{sb} - \frac{a}{b - a}e^{sa}\right)\\
    &= \log(pe^{sa} + (1 - p)e^{sb})\\
    &= \log(e^{sa}(p + (1-p)e^{sb-sa}))\\
    &= sa\frac{b-a}{b-a} + \log(p + (1-p)e^u)\\
    &= (p - 1)u + \log(p + (1-p)e^u).
\end{align*}
$$

We have that

$$
\begin{align*}
    \varphi'(u) &= (p-1) + \frac{(1-p)e^u}{p + (1 - p)e^u}
\end{align*}
$$

and that

$$
\begin{align*}
    \varphi''(u) &= \frac{p(1-p)e^u}{(p + (1 - p)e^u)^2}.
\end{align*}
$$

Notice that both $$\varphi'$$ and $$\varphi''$$ are defined everywhere. This means that $$\varphi$$ is twice-differentiable, and therefore, by Taylor's Theorem, using the Lagrange Form for the remainder, we have that for all $$u\in\mathbb{R}$$, there exists $$\xi\in\mathbb{R}$$ such that

$$
\begin{equation}
    \varphi(u) = \varphi(0) + \varphi'(0)u + \frac{1}{2}\varphi''(\xi)u^2.\quad\textbf{(3)}
\end{equation}
$$

From above, we have that $$\varphi(0) = 0$$ and that $$\varphi'(0) = 0$$. We will now use calculus to find a bound on $$\varphi''(\xi)$$. We have that

$$
\begin{align*}
    \varphi'''(u) = \frac{p(1 - p)e^u(p - (1 - p)e^u)}{(p + (1 - p)e^u)^3}.
\end{align*}
$$

Notice that $$\varphi'''(u) = 0$$ at only $$u = \log\left(\frac{p}{1 - p}\right)$$, and that it is negative to the right of that $$u$$ value and positive to the left of it. This shows that $$\varphi''(u)$$ attains its maximum value at $$u = \log\left(\frac{p}{1 - p}\right)$$. This gives us that for all $$\xi\in \mathbb{R}$$, it is the case that

$$
\begin{align*}
    \varphi''(\xi) \leq \varphi''\left(\log\left(\frac{p}{1 - p}\right)\right) = \frac{1}{4}.
\end{align*}
$$

Applying this inequality to statement $$\textbf{(3)}$$, we have that for all $$u\in \mathbb{R}$$, it is the case that

$$
\begin{align*}
    \varphi(u)\leq u^2/8.
\end{align*}
$$

But using the definition of $$\varphi(u)$$ at $$u=(b-a)s$$, we have

$$
\begin{align*}
    &\log\left(\frac{b}{b - a}e^{sb} - \frac{a}{b - a}e^{sa}\right)\leq s^2(b-a)^2/8\\
    \Longrightarrow\quad &\frac{b}{b - a}e^{sb} - \frac{a}{b - a}e^{sa}\leq e^{s^2(b-a)^2/8}.
\end{align*}
$$

Combining this with inequality $$\textbf{(2)}$$, we finally have

$$
\begin{align*}
    \mathbb{E}[e^{sV}] \leq e^{s^2(b-a)^2/8}.
\end{align*}
$$

Since $$s$$ was arbitrary, we are done.

***

Finally, we have Hoeffding's Inequality:

***

**Hoeffding's Inequality:** Let $$Z_1, \dots, Z_n$$ be independent random variables on $$\mathbb{R}$$ such that $$a_i\leq Z_i\leq b_i$$ with probability one, where $$a_i < b_i$$. If $$S_n=\sum_{i=1}^n Z_i$$, then for all $$t> 0$$, it is the case that

$$
\begin{align*}
    \textrm{P}(S_n-\mathbb{E}[S_n]\geq t)\leq e^{-2t^2/\sum(b_i-a_i)^2}
\end{align*}
$$

and

$$
\begin{align*}
    \textrm{P}(S_n-\mathbb{E}[S_n]\leq -t)\leq e^{-2t^2/\sum(b_i-a_i)^2}.
\end{align*}
$$

Note that these two bounds combine to give us

$$
\begin{align*}
    \textrm{P}(\vert S_n-\mathbb{E}[S_n]\vert\geq t)\leq 2e^{-2t^2/\sum(b_i-a_i)^2}.
\end{align*}
$$

*Proof:* Applying Chernoff's Bounding Method to the random variable $$S_n - \mathbb{E}[S_n]$$, we obtain

$$
\begin{align*}
    \textrm{P}(S_n-\mathbb{E}[S_n]\geq t)\leq \inf_{s > 0}e^{-st}\mathbb{E}\left[e^{s(S_n - \mathbb{E}[S_n])}\right]
\end{align*}
$$

By the fact that $$Z_1, \dots, Z_n$$ are independent and Hoeffding's Lemma, we have

$$
\begin{align*}
    \mathbb{E}\left[e^{s(S_n - \mathbb{E}[S_n])}\right] &= \mathbb{E}\left[e^{s\sum_{i=1}^n \left(Z_i - \mathbb{E}[Z_i]\right)}\right]\\
    &= \mathbb{E}\left[\prod_{i=1}^n e^{s(Z_i - \mathbb{E}[Z_i])}\right]\\
    &= \prod_{i=1}^n \mathbb{E}\left[e^{s(Z_i - \mathbb{E}[Z_i])}\right]\\
    &\leq \prod_{i=1}^n e^{s^2(b_i-a_i)^2/8}\\
    &= e^{(s^2/8)\sum_{i=1}^n(b_i - a_i)^2}.
\end{align*}
$$

Combining this with our inequality from above, we have

$$
\begin{align*}
    \textrm{P}(S_n-\mathbb{E}[S_n]\geq t)\leq \inf_{s > 0}e^{-st+(s^2/8)\sum_{i=1}^n (b_i-a_i)^2}.
\end{align*}
$$

Since $$s\rightarrow -st + (s^2/8)\sum_{i=1}^n(b_i-a_i)^2$$ is a parabola, we know that it attains its minimum at $$s=\frac{4t}{\sum_{i=1}^n(b_i-a_i)^2}$$. Therefore, we have

$$
\begin{align*}
    \textrm{P}(S_n-\mathbb{E}[S_n]\geq t)&\leq \inf_{s > 0}e^{-st+(s^2/8)\sum_{i=1}^n (b_i-a_i)^2}\\
    &= e^{-\left(\frac{4t}{\sum_{i=1}^n(b_i-a_i)^2}\right)t+\frac{2t^2}{\left(\sum_{i=1}^n(b_i-a_i)^2\right)^2}\sum_{i=1}^n (b_i-a_i)^2}\\
    &= e^{-2t^2/\sum_{i=1}^n(b_i-a_i)^2},
\end{align*}
$$

the first bound. Having proven the first bound, the second bound can be proven by simply applying the first bound to the random variables $$-Z_1, \dots, -Z_n$$.

***

Okay, though it took a while, we have now proven Hoeffding's Inequality. Now how do we apply it to our context? Recall that we are searching for small $$\epsilon$$ satisfying the condition that for all $$p, s\in P\times S$$, it is the case that

$$
\begin{equation*}
    P(\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert > \epsilon) \leq \frac{\delta}{\vert P\times S\vert}.
\end{equation*}
$$

Let $$X_1, \dots, X_m$$ denote the random variables representing each sample in $$X$$. By applying the definitions of $$u$$ and $$\hat{u}$$, we have that the above condition is equivalent to requiring that for all $$p, s\in P\times S$$, it be true that

$$
\begin{equation*}
    P\left(\left\vert\mathbb{E}\left[\frac{1}{m}\sum_{i=1}^m u_p(s, X_i)\right] - \frac{1}{m}\sum_{i=1}^m u_p(s, X_i)\right\vert > \epsilon\right) \leq \frac{\delta}{\vert P\times S\vert}
\end{equation*}
$$

or, after multiplying both sides of the inner inequality by $$m$$, that

$$
\begin{equation*}
    P\left(\left\vert\mathbb{E}\left[\sum_{i=1}^m u_p(s, X_i)\right] - \sum_{i=1}^m u_p(s, X_i)\right\vert > m\epsilon\right) \leq \frac{\delta}{\vert P\times S\vert}.
\end{equation*}
$$

Assume that the utility function $$u$$ is bounded, and let $$c>0$$ be such that all possible utilities lie in the interval $$[-c/2, c/2]$$. Note that this is a reasonable assumption, as in the real world, there is almost always a practical limit on the utility that can be derived from an action or state. For example, if the utility for a game involving average citizens is based on the amount of money they are able to make, it might be completely reasonable to set $$c/2$$ to be a trillion US dollars. 

For all $$p,s \in P\times S$$, applying Hoeffding's Inequality to the random variables $$u_p(s, X_1), \dots, u_p(s, X_m)$$ gives us

$$
\begin{equation*}
    P\left(\left\vert\mathbb{E}\left[\sum_{i=1}^m u_p(s, X_i)\right] - \sum_{i=1}^m u_p(s, X_i)\right\vert > m\epsilon\right) \leq 2e^{-2m^2\epsilon^2/\sum_{i=1}^m(c/2 - (-c/2))^2} = 2e^{-\frac{2m\epsilon^2}{c^2}}
\end{equation*}
$$

Now we can restrict the right-hand side of the inequality from above to be less than or equal to $$\frac{\delta}{\vert P\times S\vert}$$ and solve for $$\epsilon$$. We have

$$
\begin{align*}
    & & 2e^{-\frac{2m\epsilon^2}{c^2}} &\leq \frac{\delta}{\vert P\times S\vert}\\
    &\Longrightarrow& \epsilon &\geq \sqrt{\frac{-c^2}{2m}\log{\frac{\delta}{2\vert P\times S\vert}}}\\
    &\Longrightarrow& \epsilon &\geq c\sqrt{\frac{\log\left(2\vert P\times S\vert / \delta\right)}{2m}}.
\end{align*}
$$

Therefore, we have shown that for all $$p, s\in P\times S$$, it is the case that

$$
\begin{equation*}
    P\left(\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\vert > c\sqrt{\frac{\log\left(2\vert P\times S\vert / \delta\right)}{2m}}\right) \leq \frac{\delta}{\vert P\times S\vert},
\end{equation*}
$$

and hence, we have that

$$
\begin{equation*}
    \boxed{\textrm{P}\left(d_\infty(\Gamma_X, \Gamma_\mathscr{D})\leq c\sqrt{\frac{\log\left(2\vert P\times S\vert / \delta\right)}{2m}}\right)\geq 1 - \delta.}
\end{equation*}
$$

We have succeeded in finding PAC-style guarantees on $$d_\infty(\Gamma_X, \Gamma_\mathscr{D})$$ with bounds that are directly proportional to $$\frac{1}{\sqrt{m}}$$! This dependence on the sample size makes sense, as we expect empirical normal-form games to better approximate expected normal-form games as their sample sizes increase. One problem with these guarantees derived using Hoeffding's Inequality and the union bound is that the union bound produces unnecessarily loose bounds. This is seen in the dependence of the bounds on $$\sqrt{\log(\vert P\times S\vert)}$$. Because of this dependence, these guarantees don't scale very well to larger games. Approach 2 of the paper aims to avoid this dependence.

# Approach 2 - Rademacher Averages

Rather than finding guarantees on the deviations of individual utilities and then applying the union bound, the second approach introduced by the paper aims to use a concept called **Rademacher Averages** to directly form guarantees on the maximum deviation between utilities. We will now state the definitions of two kinds of Rademacher Averages as they have been stated in the paper.

A Rademacher distribution is a uniform distribution over the set $$\{-1, 1\}$$. Let $$\sigma\sim \textrm{Rademacher}^m$$ and for cleaner expressions, let $$\bar{I}$$ denote $$P\times S$$. We define:

$$
\begin{align*}&(1)& &\textbf{1-Draw Empirical Rademacher Average }\textrm{(1-ERA): }\\
& & &\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma):= \sup_{p, s\in \bar{I}}\left\vert\frac{1}{m}\sum_{j=1}^m\sigma_ju_p(s, x_j)\right\vert\\
&(2)& &\textbf{Rademacher Average }\textrm{(RA): }\\
& & &\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D}):= \mathbb{E}_{X, \sigma}\left[\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma)\right]
\end{align*}
$$

Note that in the paper, these definitions and the following theorems are generalized for all $$I\subseteq \bar{I}$$, but since I feel that the reason for doing so isn't apparent at this point in the Paper Walkthrough, I will leave the discussion of that for later.

In order to see how these Rademacher Averages can be applied to our context, we will first introduce a few theorems/lemmas, beginning with **Lemma A.1** whose proof will mimic that of the same lemma from the paper.

***

**Lemma A.1:** We have

$$
\begin{equation*}
    \mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right]\leq 2\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D}).
\end{equation*}
$$

*Proof:* This proof will use a technique known as **Symmetrization**. In order to use this technique, we define a second sample random variable $$X'=(x'_1, \dots, x'_m)\sim \mathscr{D}^m$$. By the definition of $$u$$ and $$\hat{u}$$, we have that

$$
\begin{equation*}
    \mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right] = \mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert \mathbb{E}_{X'}\left[\frac{1}{m}\sum_{j=1}^m u_p(s, x'_j)\right] - \frac{1}{m}\sum_{j=1}^m u_p(s, x_j)\right\vert\right].
\end{equation*}
$$

By Jensen's Inequality and Linearity of Expectation, we have

$$
\begin{align*}
    \mathbb{E}_{X}&\left[\sup_{p, s\in \bar{I}}\left\vert \mathbb{E}_{X'}\left[\frac{1}{m}\sum_{j=1}^m u_p(s, x'_j)\right] - \frac{1}{m}\sum_{j=1}^m u_p(s, x_j)\right\vert\right]\\
    &\leq \mathbb{E}_{X, X'}\left[\sup_{p, s\in \bar{I}}\left\vert \frac{1}{m}\sum_{j=1}^m u_p(s, x'_j) - \frac{1}{m}\sum_{j=1}^m u_p(s, x_j)\right\vert\right]\\
    &= \mathbb{E}_{X, X'}\left[\sup_{p, s\in \bar{I}}\left\vert \frac{1}{m}\sum_{j=1}^m \left(u_p(s, x'_j) - u_p(s, x_j)\right)\right\vert\right]
\end{align*}
$$

Since $$x_1, \dots, x_m, x'_1, \dots x'_m$$ are all independently and identically distributed random variables, for each $$p, s\in \bar{I}$$ and index $$j$$, the distribution of $$u_p(s, x'_j) - u_p(s, x_j)$$ is unchanged after being multiplied by a Rademacher variable. This gives us

$$
\begin{align*}
    \mathbb{E}_{X, X'}&\left[\sup_{p, s\in \bar{I}}\left\vert \frac{1}{m}\sum_{j=1}^m \left(u_p(s, x'_j) - u_p(s, x_j)\right)\right\vert\right]\\
    &= \mathbb{E}_{X, X', \sigma}\left[\sup_{p, s\in \bar{I}}\left\vert \frac{1}{m}\sum_{j=1}^m \sigma_j\left(u_p(s, x'_j) - u_p(s, x_j)\right)\right\vert\right].
\end{align*}
$$

By Triangle Inquality and Linearity of Expectation, we have

$$
\begin{align*}
    \mathbb{E}_{X, X', \sigma}&\left[\sup_{p, s\in \bar{I}}\left\vert \frac{1}{m}\sum_{j=1}^m \sigma_j\left(u_p(s, x'_j) - u_p(s, x_j)\right)\right\vert\right]\\
    &\leq \mathbb{E}_{X', \sigma}\left[\sup_{p, s\in \bar{I}}\left\vert \frac{1}{m}\sum_{j=1}^m \sigma_ju_p(s, x'_j)\right\vert\right] + \mathbb{E}_{X, \sigma}\left[\sup_{p, s\in \bar{I}}\left\vert -\frac{1}{m}\sum_{j=1}^m \sigma_ju_p(s, x_j)\right\vert\right]\\
    &= 2\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D})
\end{align*}
$$

***

Next, we introduce a theorem known as **McDiamard's Bounded Difference Inequality**. The proof of this theorem is beyond the scope of this Paper Walkthrough, but several proofs exist online. Notice, however, that McDiarmard's Bounded Difference Inequality is a generalization of Hoeffding's Inequality where $$S_n$$ is replaced with an arbitrary function of the provided independent random variables that satisfies a "bounded difference condition".

***

**McDiamard's Bounded Difference Inequality** [4]**:** Let $$X_1, \dots, X_l$$ be independent random variables and let $$h(x_1, \dots, x_l)$$ be a function s.t. a change in variable $$x_i$$ can change the value of the function by no more than $$c_i$$:

$$
\begin{equation*}
    \sup_{x_1, \dots, x_l, x'_i}\vert h(x_1, \dots, x_i, \dots, x_l) - h(x_1, \dots, x'_i, \dots, x_l)\vert \leq c_i.\quad\textrm{(Bounded Difference Condition)}
\end{equation*}
$$

Then, for any $$\epsilon > 0$$, we have

$$
\begin{equation*}
    \textrm{P}(h(X_1, \dots, X_l) - \mathbb{E}\left[h(X_1, \dots, X_l)\right] > \epsilon) \leq \textrm{exp}\left(-2\epsilon^2/\sum_{i=1}^l c_i^2\right).
\end{equation*}
$$

***

Combining Lemma A.1 and McDiamard's Bounded Difference Inequality, we can now use Rademacher Averages to derive the guarantees we are looking for. Let $$X_1, \dots, X_m$$ denote the random variables representing each sample in $$X$$, and let $$\sigma_1, \dots, \sigma_m$$ denote each component random variable of $$\sigma$$. First notice that by the Linearity of Expectation, we have that

$$
\begin{align*}
    & & &\mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right] - 2\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D})\\
    &= & &\mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right] - 2\mathbb{E}_{X, \sigma}\left[\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma)\right]\\
    &= & &\mathbb{E}_{X, \sigma}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert - 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma)\right]\\
    &= & &\mathbb{E}_{X_1, \dots, X_m,\\ \sigma_1, \dots, \sigma_m}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \frac{1}{m}\sum_{j=1}^m u_p(s, X_j)\right\vert - 2\sup_{p, s\in \bar{I}}\left\vert\frac{1}{m}\sum_{j=1}^m\sigma_ju_p(s, X_j)\right\vert\right]\quad\textbf{(4)}
\end{align*}
$$

 Consider the function $$h$$ defined by

$$ h(X_1,\dots, X_m, \sigma_1, \dots, \sigma_m):= \sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \frac{1}{m}\sum_{j=1}^m u_p(s, X_j)\right\vert - 2\sup_{p, s\in \bar{I}}\left\vert\frac{1}{m}\sum_{j=1}^m\sigma_ju_p(s, X_j)\right\vert$$

Does $$h$$ satisfy the bounded difference condition? As we did in the discussion of Hoeffding's Inequality, we will assume that all utilities are bounded to some interval $$[-c/2, c/2]$$. If, for some index $$J$$, the input $$X_J$$ is modified, we have that each of the $$u_p(s, X_J)$$ terms will change by at most $$c$$, and as a result, the values of $$\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \frac{1}{m}\sum_{j=1}^m u_p(s, X_j)\right\vert$$ and $$\sup_{p, s\in \bar{I}}\left\vert\frac{1}{m}\sum_{j=1}^m\sigma_ju_p(s, X_j)\right\vert$$ will both change by at most $$c/m$$. This means that for a change in one of the $$X_j$$'s, the output of $$h$$ can change by at most $$3c/m$$. If, for some index $$J'$$, the input $$\sigma_{J'}$$ is modified, we have that each of the $$\sigma_{J'}u_p(s, X_{J'})$$ terms will change by at most $$c$$, and as a result, the term $$\sup_{p, s\in \bar{I}}\left\vert\frac{1}{m}\sum_{j=1}^m\sigma_ju_p(s, X_j)\right\vert$$ will both change by at most $$c/m$$. This means that for a change in one of the $$\sigma_j$$'s, the output of $$h$$ can change by at most $$2c/m$$.

Therefore, we have shown that $$h$$ does satisfy the bounded difference condition, with $$c_i = 3c/m$$ for each $$i$$ corresponding to one of the $$X_j$$'s and $$c_i = 2c/m$$ for each $$i$$ corresponding to one of the $$\sigma_j$$'s.

From this point onwards, we will use $$h(X, \sigma)$$ to denote $$h(X_1, \dots, X_m, \sigma_1, \dots \sigma_m)$$. Notice that

$$h(X, \sigma) = \sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert - 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma)$$

and that by $$\textbf{(4)}$$, we have that

$$\mathbb{E}_{X, \sigma}\left[h(X, \sigma)\right] = \mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right] - 2\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D})$$

We can now apply McDiamard's Inequality to the function $$h$$, giving us the following result

$$
\begin{align*}
    \textrm{P}(h(X, \sigma) - \mathbb{E}_{X, \sigma}\left[h(X, \sigma)\right] > \epsilon) &\leq \textrm{exp}\left(-2\epsilon^2/\sum_i c_i^2\right)\\
    &= \textrm{exp}\left(-2\epsilon^2/\left(m\cdot\left(\frac{3c}{m}\right)^2 + m\cdot\left(\frac{2c}{m}\right)^2\right)\right)\\
    &= \textrm{exp}\left(\frac{-2m\epsilon^2}{13c^2}\right)
\end{align*}
$$

Using the complement rule of probability, we have

$$\textrm{P}(h(X, \sigma) - \mathbb{E}_{X, \sigma}\left[h(X, \sigma)\right] \leq \epsilon) \geq  1 - \textrm{exp}\left(\frac{-2m\epsilon^2}{13c^2}\right)$$

Since Lemma A.1 implies that $$\mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right] - 2\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D})\leq 0$$, we have that

$$
\begin{align*}
    & & h(X, \sigma) - \mathbb{E}_{X, \sigma}\left[h(X, \sigma)\right] &\leq \epsilon\\
    &\Longrightarrow & \left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert - 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma)\right] - \quad\quad\quad\quad\quad &\\
    & & \left[\mathbb{E}_{X}\left[\sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert\right] - 2\mathfrak{R}_m(\Gamma, \bar{I}, \mathscr{D})\right] &\leq \epsilon\\
    &\Longrightarrow & \sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert - 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma) & \leq \epsilon\\
    &\Longrightarrow & \sup_{p, s\in \bar{I}}\left\vert u_p(s; \mathscr{D}) - \hat{u}_p(s; X)\right\vert &\leq 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma) + \epsilon\\
    &\Longrightarrow & d_\infty(\Gamma_X, \Gamma_\mathscr{D}) &\leq 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma) + \epsilon.
\end{align*}
$$

Therefore, we now have that

$$\textrm{P}(d_\infty(\Gamma_X, \Gamma_\mathscr{D}) \leq 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma) + \epsilon) \geq  1 - \textrm{exp}\left(\frac{-2m\epsilon^2}{13c^2}\right)$$

Setting the right side of the inequality to be greater than or equal to $$1-\delta$$ and then solving for $$\epsilon$$ gives us

$$\epsilon \geq \sqrt{13}c\sqrt{\frac{\ln(1/\delta)}{2m}}.$$

Therefore, we have succeeded in using Rademacher Averages to form this second PAC-style guarantee on $$d_\infty(\Gamma_X, \Gamma_\mathscr{D})$$:

$$\boxed{\textrm{P}\left(d_\infty(\Gamma_X, \Gamma_\mathscr{D}) \leq 2\hat{\mathfrak{R}}_m^1(\Gamma, \bar{I}, X, \sigma) + \sqrt{13}c\sqrt{\frac{\ln(1/\delta)}{2m}}\right) \geq  1 - \delta.}$$

The bound derived here is slightly looser than the bound derived in the paper. In the paper, the $$\sqrt{13}$$ constant is replaced with a $$3$$. I am not entirely sure how the slightly tighter bound in the paper was derived, but I will be emailing Dr. Greenwald and asking about it. If I get an answer, I'll update this part of the Paper Walkthrough. For analysis purposes, however, this bound is very similar to the one in the paper.

Unlike the bounds derived from Hoeffding's Inequality, these bounds have no explicit dependence on the number of elements in $$\bar{I}=P\times S$$. This means that this guarantee can scale well to arbitrarily large games. These bounds are, however, data-dependent and require a sample to be drawn from the distribution. Unlike the bounds derived from Hoeffding's Inequality, these bounds cannot be calculated *a priori*.

# Conclusion

We have derived PAC-style guarantees on $$d_\infty(\Gamma_X, \Gamma_\mathscr{D})$$ using two approaches. The first approach based on Hoeffding's Inequality produces bounds that have a factor that is $$O(\sqrt{\log(\vert P\times S\vert)})$$, making the resulting guarantees not scale well to large games. The second approach resolves this issue and has no such factor, but requires a sample to first be drawn to produce guarantees for that specific sample. In part 3 of this Paper Walkthrough, we will implement two different algorithms discussed in the paper for using empirical normal-form games to approximate expected normal-form games, and then use various experiments to analyze how our two different guarantees compare in various different contexts.

# References

[1] Viqueira, Enrique Areyan, et al. “Learning Equilibria of Simulation-Based Games.” ArXiv:1905.13379 [Cs], May 2019. arXiv.org, <http://arxiv.org/abs/1905.13379>.

[2] E. Weisstein, “Bonferroni Inequalities,” Wolfram MathWorld. Available: [https://mathworld.wolfram.com/BonferroniInequalities.html](https://mathworld.wolfram.com/BonferroniInequalities.html). [Accessed: 02-Jul-2021].

[3] C. Scott and A. Zimmer, “Hoeffding's Inequality,” EECS 598: Statistical Learning Theory, Winter 2014. Available: [http://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/03_hoeffding.pdf](http://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/03_hoeffding.pdf). [Accessed: 02-Jul-2021].

[4] M. Riondato, “Rademacher Averages: Theory and Practice,” Two Sigma. Available: [https://www.twosigma.com/wp-content/uploads/Riondato_-_RademacherTheoryPractice_-_Dagstuhl_-_Slides.pdf](https://www.twosigma.com/wp-content/uploads/Riondato_-_RademacherTheoryPractice_-_Dagstuhl_-_Slides.pdf). [Accessed: 04-Jul-2021].