# Scala
It contains the information &amp; sources of Scala, which is a programming language.

## At the beginning

This is the first page.


## Install 

```sh
sudo apt-get install scala
```

In case of ```Failed to initialize compiler: object java.lang.Object in compiler mirror not found```

use ```openjdk-8``` rather than ```java-9-openjdk-amd64```
```sh
sudo update-alternatives --config java
sudo update-alternatives --config javac

There are 2 choices for the alternative java (providing /usr/bin/java).

  Selection    Path                                            Priority   Status
------------------------------------------------------------
  0            /usr/lib/jvm/java-9-openjdk-amd64/bin/java       1091      auto mode
* 1            /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java   1081      manual mode
  2            /usr/lib/jvm/java-9-openjdk-amd64/bin/java       1091      manual mode

```
