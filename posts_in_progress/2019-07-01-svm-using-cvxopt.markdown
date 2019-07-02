---
layout: post
title:  "SVM Implementation using CVXOPT - Python"
date:   2019-07-01
categories: [machine learning]
---

Over the past couple of days, I've been spending the majority of my time really learning the theory behind Support Vector Machines (SVMs). I've come across many useful resources including the [MIT OCW video](https://www.youtube.com/watch?v=_PwhiWxHK8o) on the subject and the almighty Wikipedia for learning about concepts like [Karush-Kuhn-Tucker Conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions), [Wolfe Duality](https://en.wikipedia.org/wiki/Wolfe_duality), and more. I would personally recommend checking out all of those links, as the MIT video provides a nice walkthrough for the math of SVMs and the Wikipedia links clarify the inner workings of that math.

*Side Note:* While the MIT video is a great resource and was really useful in getting a feel for the math, the professor does make a mistake when applying Lagrange Multipliers. He sets the constraint for the function to be $y_i(\vec{w}\cdot \vec{x_i} + b) - 1 = 0$. Obviously, this isn't necessarily true for all $i$ as it is not necessary for all points to be on the margin of the SVM. The constraint is actually the inequality, $y_i(\vec{w}\cdot \vec{x_i} + b) - 1 \geq 0$. That's the reason KKT conditions are applied to the problem, and it's the reason the professor applies the constraint, $\lambda_i \geq 0$.

After developing somewhat of an understanding of the algorithm, my first project was to create an actual implementation of the SVM algorithm. Though it didn't end up being entirely from scratch as I used CVXOPT to solve the convex optimization problem, the implementation helped me better understand how the algorithm worked and ... . In this post, I hope to walk you through that implementation. Note that this post assumes an understanding of the underlying math behind SVMs. If you feel uncomfortable on that front, I would again recommend checking out the resources linked above.

Anyways, with that out of the way, let's get into it! Let's start by writing out