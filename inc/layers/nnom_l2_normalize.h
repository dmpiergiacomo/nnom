#ifndef __NNOM_L2NORM_H__
#define __NNOM_L2NORM_H__

#include "nnom.h"


// L2 Normalize
typedef struct _nnom_l2_normalize_config_t
{
	nnom_layer_config_t super;
} nnom_l2_normalize_config_t;

typedef struct _nnom_l2_normalize_layer_t
{
	nnom_layer_t super;
} nnom_l2_normalize_layer_t;

// method
nnom_status_t l2norm_run(nnom_layer_t *layer);
nnom_status_t l2norm_build(nnom_layer_t *layer);
nnom_status_t l2norm_free(nnom_layer_t *layer);

// API
nnom_layer_t *l2norm_s(const nnom_layer_config_t * config);
nnom_layer_t *L2Norm(void);

#endif /* __NNOM_L2NORM_H__ */
