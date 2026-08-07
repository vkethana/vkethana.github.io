---
layout: base
title: Home
---

<h1 class="page-heading">Posts</h1>

<div class="tag-filter" id="tag-filter" role="group" aria-label="Filter posts by tag">
  <button type="button" class="tag-chip is-active" data-tag="all">All</button>
  <button type="button" class="tag-chip" data-tag="machine learning">machine learning</button>
  <button type="button" class="tag-chip" data-tag="linguistics">linguistics</button>
  <button type="button" class="tag-chip" data-tag="school project">school project</button>
</div>

<p class="tag-filter-empty" id="tag-filter-empty" hidden>No posts with this tag.</p>

<div class="post-list" id="post-list">
  {% assign published_posts = site.posts | where_exp: "post", "post.published != false" %}
  {% for post in published_posts %}
    <article class="post-card" data-tags="{{ post.tags | join: '|' | escape }}">
      <a class="post-card-thumb" href="{{ post.url | prepend: site.baseurl }}" aria-label="{{ post.title }}">
        {% if post.featured_image %}
          <img src="{{ post.featured_image }}" alt="">
        {% else %}
          <span class="post-card-fallback">{{ post.title | slice: 0 }}</span>
        {% endif %}
      </a>
      <div class="post-card-body">
        <h2><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h2>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
      </div>
    </article>
  {% endfor %}
</div>

<script src="/assets/js/tag-filter.js" defer></script>
