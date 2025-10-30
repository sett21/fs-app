## Генеративный режим (GPU)

Сервис `gen` (порт 8081) принимает 3 файла: `selfie`, `postcard`, `face`.

**Требование к селфи:** на изображении должен быть зелёный прямоугольник цвета `#00ff84` в том месте и пропорции, где должна оказаться открытка.

Запуск:
```bash
docker compose up --build
```

### Очистка Docker (когда кончается место)

Ниже — безопасный чек‑лист. Команды удаляют неиспользуемые образы, кэш сборки и тома. Если какие‑то контейнеры/образы вам нужны, проверьте вывод перед удалением.

Проверить, что занимает место:
```bash
docker system df
df -h
```

Быстрая общая очистка (образы, кэш, тома, сети без использования):
```bash
docker system prune -af --volumes
```

Отдельно подчистить кэш билдера (BuildKit):
```bash
docker builder prune -af
```

Очистить «висячие» образы и тома (если нужно отдельно):
```bash
docker image prune -af
docker volume prune -f
```

Удалить том кэша HuggingFace, если он больше не нужен (имя смотрим в списке):
```bash
docker volume ls | grep hf_cache
docker volume rm <имя_тома_hf_cache>
```

Удалить отладочные файлы внутри запущенного контейнера `gen`:
```bash
docker compose -f docker-compose.gpu.yml exec gen sh -lc 'rm -f /tmp/debug_*'
```

Подсказка: если хотите автоматизировать ежедневную очистку, можно настроить systemd‑таймер, запускающий
`docker system prune -af --volumes` и `docker builder prune -af` раз в сутки.
