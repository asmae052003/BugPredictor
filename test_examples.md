# Test Examples for Bug Predictor

Here are some code snippets you can use to test the application.

## C++ Examples

### 1. Simple Clean Code (Low Complexity)
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### 2. Complex Code (Higher Risk of Bugs)
This code has higher cyclomatic complexity and Halstead volume.
```cpp
#include <iostream>

void complexFunction(int a, int b, int c) {
    if (a > 0) {
        if (b > 0) {
            for (int i = 0; i < c; i++) {
                if (i % 2 == 0) {
                    std::cout << "Even" << std::endl;
                } else {
                    std::cout << "Odd" << std::endl;
                }
            }
        } else {
            while (a > 0) {
                a--;
                if (a == 5) break;
            }
        }
    } else if (c < 0) {
        switch (b) {
            case 1: std::cout << "One"; break;
            case 2: std::cout << "Two"; break;
            default: std::cout << "Other";
        }
    }
}
```

## Java Examples

### 1. Simple Clean Code
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

### 2. Complex Code (Higher Risk)
```java
public class ComplexLogic {
    public void process(int x, int y, int[] data) {
        try {
            if (x > 10 && y < 5) {
                for (int i = 0; i < data.length; i++) {
                    if (data[i] > 0) {
                        if (data[i] % 2 == 0) {
                            System.out.println("Positive Even");
                        } else {
                            System.out.println("Positive Odd");
                        }
                    } else if (data[i] < 0) {
                        System.out.println("Negative");
                    } else {
                        System.out.println("Zero");
                    }
                }
            } else {
                switch (x) {
                    case 1: y = y + 1; break;
                    case 2: y = y + 2; break;
                    default: y = 0;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
