export const log = (content: string, color = 'red', size = '1.2em') => {
  console.log(`%c${content}:`, `color:${color};font-size:${size}`)
}
