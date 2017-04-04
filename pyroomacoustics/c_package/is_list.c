
#include "room.h"

#include <stdlib.h>

void is_list_insert(is_ll_t **head, is_ll_t *new_node)
{
  /*
   * Add a new node to the linked list of visible sources
   */
  if (*head == NULL)
  {
    *head = new_node;
    new_node->next = NULL;
  }
  else
  {
    new_node->next = *head;
    *head = new_node;
  }
}

void is_list_pop(is_ll_t **head)
{
  /* Delete first element in linked list */
  is_ll_t *tmp = (*head)->next;

  if ((*head)->visible_mics != NULL)
    free((*head)->visible_mics);

  free(*head);

  *head = tmp;
}

void is_list_delete(is_ll_t **head)
{
  /*
   * Deletes the linked list of image sources
   * and frees the memory
   */

  while (*head != NULL)
    is_list_pop(head);
}

int is_list_count(is_ll_t *node)
{
  if (node == NULL)
    return 0;
  else
    return is_list_count(node->next) + 1;
}

void is_list_print(is_ll_t *node, int dim)
{
  /* print the linked list */
  if (node != NULL)
  {
    is_list_print(node->next, dim);
    print_vec(node->is.loc, dim);
  }
}

