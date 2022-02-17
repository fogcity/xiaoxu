export const log = (target: { title?: any; content?: any }, color?: string, fontSize?: string) => {
  const { title, content } = target
  if (title) console.log(title, `%c${content}`, `color:${color};font-size:${fontSize}`)
  else console.log(`%c${content}`, `color:${color};font-size:${fontSize}`)
}
