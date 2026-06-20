# Calliope 10M vs 30M Baseline Report

## Summary

- 10M source: `experiments\calliope_10m_20260620_161343`
- 30M source: `experiments\Calliope-30M-run001`
- prompts compared: 3
- samples compared: 12

## Final Metrics

| Model | Iter | Train Loss | Val Loss | Loss EMA | VRAM GB |
|---|---:|---:|---:|---:|---:|
| 10M | 10000 | 1.8444 | 1.8575 | 1.8465 | 1.714 |
| 30M | 10000 | 1.6505 | 1.6788 | 1.6565 | 1.370 |

## Readout

- 30M has lower validation loss on this run.
- 30M samples are generally more coherent and stay closer to TinyStories style.
- Both models still show repetition and occasional broken text, so the next useful step is more fixed prompts plus a stricter validation/sample rubric.

## Side-by-Side Samples

### `Lily found a tiny` sample 0

**10M**

Lily found a tiny toy car. She wanted to play with it, but it was too small. So, she asked her mommy if she could play with it. Her mommy said no, but she wouldn't let Sarah play with it. She was upset and didn't want to play with Sarah.

On the day of the game, Lily and her mommy went to a stadium to watch the stadium. Lily was happy to spend time with her mommy and they played together and had lots of fun. Lily realized that it was important to listen to her parents and not touch things that were not toys

**30M**

Lily found a tiny toy car. She wanted to play with it, but it was too big for her. She asked her mom, "Can I play with the car?"

"No, Lily, this is your car. You can't play with it for a little while. It is not a toy. It is a modern toy."

Lily felt sad. She really wanted to play with the car. She asked her mom, "Can I play with the car, please?"

Her mom said, "No, Lily. You can play with something else. You can take turns and

### `Lily found a tiny` sample 1

**10M**

Lily found a tiny bug crawling around her yard, and she was very happy. She told her mommy about the bug and how it could keep her life. Her mommy smiled and said, "Yes, it's nice to meet you too." And they continued playing in the rain, happy that Lily had a new friend to keep her company.<|endoftext|>One day, a little boy named Tim found a big, shiny silver coin on the ground. He picked it up and showed it to his mom. "Look, Mommy! I found a coin!" said Tim. "It's so pretty!" said his mom

**30M**

Lily found a tiny bug and gently picked it up. She was so happy with her new friend. She put the bug on her finger and went outside. From that day on, Lily and her new friend went to the park together.<|endoftext|>Once upon a time, there was a little girl named Lily. She loved to play with her toy boat in the bathtub. One day, she made a big splash with her sail and it went into the bathtub. She giggled and splashed around, trying to make it go away.

Suddenly, her mom came in and said, "Lily, why

### `Lily found a tiny` sample 2

**10M**

Lily found a tiny mouse who loved to play with. She brought the mouse inside to show her mommy.

"Look, Mommy! I found a mouse!" Lily said, pointing to a small mouse who was hiding behind a tree. "Wow, that is a big mouse! Can I be your friend?"

"Sure, Lily. That's very kind of mouse. I would like to play with you," her mommy said with a smile.

Lily and her mommy played together for a while, but then Lily had an idea. "Let's make a big potato soup

**30M**

Lily found a tiny mouse in the attic. It was very small, and it looked very sad. Lily decided to help the mouse. She took it in her arms and carried it around the attic until she found a box. She put the mouse inside and said, "Let's be friends!" The mouse was happy to have a new friend. Lily and the mouse played together all day. They had lots of fun and became best friends.<|endoftext|>Once upon a time, there was a playful little bunny. The bunny loved to hop around and explore the world. One day, the bunny saw a pile of leaves on the

### `Lily found a tiny` sample 3

**10M**

Lily found a tiny bug and showed it to her mom. "Look, Mommy! A bug!" she said. "Ew, that's sillyie. Let me tell you where it is."

Lily went to her table and said, "Wow, it's so cool! Here, this bug is a shiny bug that it is bright and brown." Her mom replied, "That's a good idea, Lily. But first, let's go to the living room and find something to eat." Lily was happy and said, "OK, mommy. Let's go."

At the

**30M**

Lily found a tiny bug and showed it to her mom. "Look, mommy! I found a bug!" she said. "Wow, that's an attractive bug!" her mom replied. Lily was happy that her mom liked her adventure and asked if she could see the bug too.<|endoftext|>Once upon a time, there was a little girl named Lily. She loved to dance and twirl around in her pretty dress. One day, she was dancing in a park when she saw a big red ball. She wanted to play with it, but it was too heavy for her to lift.

Lily started

### `Once upon a time` sample 0

**10M**

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she was playing on her porch when she saw an unusual old glass. She went to take a closer look and saw that it was very old and had a big hole in it.

Lily was curious and decided to look inside the dead glass. She didn't know what it was, but she believed it was a little seed. She carefully opened the jar and found a tiny seed inside. The seed started to grow and bloom. Lily's mommy was very proud of her.

**30M**

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she was playing with her toys when she found an unusual rock. She had never seen anything like it before.

Lily put the rock in her mouth to see if anything happened. It was very strange. She didn't know what it felt to be because it was different from her. She told her mom about it and her mom said it was called a special rock.

Lily didn't like the thing and thought it was ugly. She started to cry more and she felt

### `Once upon a time` sample 1

**10M**

Once upon a time there was an old lady who was very humble. She liked to take care of things. One day she decided to go for a walk and he was so excited.

At home, the old man took out a small bowl of ice cream. He showed the old lady how to scoop out some ice cream. She was very generous with the treat. The old man took the sweet ice cream down to the ground and began to eat it. He was very pleased to have eaten all the ice cream.

The old man was so happy that he bought the freezer for himself to take with him

**30M**

Once upon a time there was an alligator who lived in the sea. He loved to explore and he was always looking for new adventures. One day, he came across a very special cave. It was full of colorful rocks and rocks. He decided he wanted to explore what was inside.

But then, something different happened. He found a rock that was very soft and inviting. He decided to investigate it. When he opened the door, he saw a big fish! The fish was so excited and he wanted to explore.

He swam around the cave and said, â€œDo

### `Once upon a time` sample 2

**10M**

Once upon a time there was a small rabbit who was very lonely. He was lonely and wanted to find someone to play with.

One day he saw a 3-year-old girl. She was very compassionate and wanted to help her so she said yes.

The little girl asked the rabbit if she could help. The rabbit said yes, so the little girl helped him up and they both took turns using the little boy's hand.

The little girl was so happy. She thanked the rabbit and shared her love with the little girl. They became good friends and the little girl was always kind

**30M**

Once upon a time there was a small rabbit who was feeling very lonely. He wanted something to make him happy. So he went to the other side of the forest and found some sweet berries. He was so happy! He ate them and then he felt even happier.

The rabbit was so pleased with his new discovery that he decided to share it with everyone. He ran around the forest, exploring all the different things there. He saw a little duck and a frog who were both very happy.

The rabbit shared his meal with the frog and they both became very good friends. They talked and shared stories

### `Once upon a time` sample 3

**10M**

Once upon a time, there was a rabbit named Benny. Benny loved to play and have fun. One day, Benny was playing with his toys and he accidentally broke a vase. He felt bad and decided to take a nap. 

As Benny woke up, he saw that the vase was broken. He picked it up and showed it to his friend, a little bird named Mittens. "Look what happened, Mittens?" Tweet asked. "I broke it," said the bird. "Don't worry, Tweet. Let's go home and play again." 

As they were leaving

**30M**

Once upon a time, there was a little boy named Timmy. Timmy had a red car that he loved to drive around the park. One day, Timmy's mommy gave him a big bag of candy from the store. 

Timmy was so happy and he took the candy with him everywhere he went. But then, his little sister came and wanted some too. Timmy didn't want to share his candy, so he tried to take the bag away from his sister. 

His sister got upset and started to yell again. Suddenly, Timmy felt guilty and decided to give

### `The little dog wanted` sample 0

**10M**

The little dog wanted to play with the little dog, but it was too scared to move. He did not want to be friends with the little girl. He told her to be quiet and not bother her.

Her mom said, "Don't worry, we can pet your dog. He is friendly and likes to play. Come, let's go." The little dog let go of the little girl. He was surprised and happy. He did not want to be alone.

The little girl and her mom walked to the little dog. The little dog saw the little boy. The little boy smiled and

**30M**

The little dog wanted to play with the ball too. He ran around and had so much fun.

Finally, he stopped at a store. He saw lots of fun things to do. He asked the owner if he could take a turn and the owner said yes.

The little dog was very happy. He had a new friend. He named the ball and they played together all day. And they all lived happily ever after.<|endoftext|>Once upon a time, there was a little girl named Lily. She loved to play hide-and-seek with her friends. One day, while playing hide-and

### `The little dog wanted` sample 1

**10M**

The little dog wanted to play with the other animals. They were very competitive and wanted to get to each other.

"Let's play too!" the little dog said. "We can play with the other animals and have fun!"

"OK," said the little dog. They played with the other animals for a while, but then they saw something strange. The little dog ran to get a better look.

The other animals were happy to see the little dog. They could not stop looking and play, and they had a big hug. The little dog and his friends had a good day.

**30M**

The little dog wanted to play with the ball, but he did not know how to stop. Suddenly, he heard a voice. It was the neighbor's dog. He was angry and surprised. "Why don't you play with my ball instead?" he said.

The little dog was happy and said yes. He played with the ball all day. The neighbor saw what he did and felt bad. She told him not to be so mean. From that day on, the neighbor did not like the neighbor's noise, because he kept a big and strong dog.<|endoftext|>Once upon a time, there was a

### `The little dog wanted` sample 2

**10M**

The little dog wanted to have the best time with the mule. The mule didn't know what to do. So, he used his strong wings to fly away. The mule was very happy and the little dog was happy too.<|endoftext|>Once upon a time, there was a little pig named Snowman. Snowman was very happy because he loved to play with his ball. One day, Snowman saw a big stick and went to play with it. "Can I play with the stick?" said Snowy. "Of course," said Snowman. 

As they played, grandma came to

**30M**

The little dog wanted to have fun too! So, he asked the little dog to come with him. But the little dog did not want to go. He said, "No, I don't want to be your friend. I want to play with my ball." The big dog got angry and started to chase the little dog. The little dog ran and ran, but the dog was fast.

The dog chased the little dog and hit a wall. The wall broke and the little dog could not get out. The little dog was sad and went back to his hole. He learned that not everything that looks

### `The little dog wanted` sample 3

**10M**

The little dog wanted the tree to be safe. He ran up to the tree and asked, "Can I have the tree, please?"

The tree replied, "Yes, but be careful of the tree. It's very big and it might hurt you." The little dog was so happy and thanked the tree.

The tree said, "Thank you! You are very good and kind." The little dog felt happy and safe with the tree. From that day on, the little dog and the tree were best friends. They both lived happily ever after.<|endoftext|>Once upon a time, there was a

**30M**

The little dog wanted to play, but he was too big for that. He was very small, so he had to move and move. He tried to move by himself, but he was too big. He tried to stand, but he could not move. He was very sad.

He asked a bird for help. The bird said, "I can help you work." The bird flew down and took the stick off. The bird was so happy to have it back. He said, "Thank you, bird!" They both looked at each other and talked. They became good friends.

The man
