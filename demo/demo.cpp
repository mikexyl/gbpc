#include <SFML/Graphics.hpp>

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Animation");
    sf::CircleShape point(5);
    point.setFillColor(sf::Color::Red);

    sf::Vector2f startPos(100, 100);
    sf::Vector2f endPos(700, 500);
    sf::Vector2f currentPos = startPos;
    float t = 0;
    float freq = 30.0;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        if (t < 1) {
            t += 0.01;
            currentPos = startPos + t * (endPos - startPos);
            point.setPosition(currentPos);
        }

        window.clear();
        window.draw(point);
        window.display();

        sf::sleep(sf::milliseconds(1000 / freq));
    }

    return 0;
}
