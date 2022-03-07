module.exports = {
  // site config
  lang: "en-US",
  title: "Cryptoart Cookbook",
  description: "This is my first VuePress site",

  // theme and its config
  theme: "@vuepress/theme-default",
  themeConfig: {
    logo: "https://vuejs.org/images/logo.png",
    sidebar: [
      {
        text: "Preface",
        link: "preface.md",
      },
      {
        text: "Chapter 1",
        children: [
          {
            text: "One",
            link: "chapter1.md#one",
          },
          {
            text: "Two",
            link: "chapter1.md#two",
          },
        ],
      },
    ],
  },
};
